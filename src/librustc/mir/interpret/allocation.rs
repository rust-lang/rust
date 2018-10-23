// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The virtual memory representation of the MIR interpreter

use super::{
    UndefMask,
    Relocations,
    EvalResult,
    Pointer,
    AllocId,
    Scalar,
    ScalarMaybeUndef,
    write_target_uint,
    read_target_uint,
    truncate,
};

use std::ptr;
use ty::layout::{self, Size, Align};
use syntax::ast::Mutability;

/// Classifying memory accesses
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAccess {
    Read,
    Write,
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct Allocation<Tag=(),Extra=()> {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer
    pub bytes: Vec<u8>,
    /// Maps from byte addresses to extra data for each pointer.
    /// Only the first byte of a pointer is inserted into the map; i.e.,
    /// every entry in this map applies to `pointer_size` consecutive bytes starting
    /// at the given offset.
    pub relocations: Relocations<Tag>,
    /// Denotes undefined memory. Reading from undefined memory is forbidden in miri
    pub undef_mask: UndefMask,
    /// The alignment of the allocation to detect unaligned reads.
    pub align: Align,
    /// Whether the allocation is mutable.
    /// Also used by codegen to determine if a static should be put into mutable memory,
    /// which happens for `static mut` and `static` with interior mutability.
    pub mutability: Mutability,
    /// Extra state for the machine.
    pub extra: Extra,
}

trait AllocationExtra<Tag> {
    /// Hook for performing extra checks on a memory access.
    ///
    /// Takes read-only access to the allocation so we can keep all the memory read
    /// operations take `&self`.  Use a `RefCell` in `AllocExtra` if you
    /// need to mutate.
    fn memory_accessed(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        access: MemoryAccess,
    ) -> EvalResult<'tcx> {
        Ok(())
    }
}

impl<Tag, Extra: Default> Allocation<Tag, Extra> {
    /// Creates a read-only allocation initialized by the given bytes
    pub fn from_bytes(slice: &[u8], align: Align) -> Self {
        let mut undef_mask = UndefMask::new(Size::ZERO);
        undef_mask.grow(Size::from_bytes(slice.len() as u64), true);
        Self {
            bytes: slice.to_owned(),
            relocations: Relocations::new(),
            undef_mask,
            align,
            mutability: Mutability::Immutable,
            extra: Extra::default(),
        }
    }

    pub fn from_byte_aligned_bytes(slice: &[u8]) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1, 1).unwrap())
    }

    pub fn undef(size: Size, align: Align) -> Self {
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        Allocation {
            bytes: vec![0; size.bytes() as usize],
            relocations: Relocations::new(),
            undef_mask: UndefMask::new(size),
            align,
            mutability: Mutability::Mutable,
            extra: Extra::default(),
        }
    }
}

impl<'tcx> ::serialize::UseSpecializedDecodable for &'tcx Allocation {}

/// Byte accessors
impl<'tcx, Tag, Extra: AllocationExtra<Tag>> Allocation<Tag, Extra> {
    /// The last argument controls whether we error out when there are undefined
    /// or pointer bytes.  You should never call this, call `get_bytes` or
    /// `get_bytes_with_undef_and_ptr` instead,
    ///
    /// This function also guarantees that the resulting pointer will remain stable
    /// even when new allocations are pushed to the `HashMap`. `copy_repeatedly` relies
    /// on that.
    fn get_bytes_internal(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align,
        check_defined_and_ptr: bool,
    ) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_align(ptr.into(), align)?;
        self.check_bounds(ptr, size, true)?;

        if check_defined_and_ptr {
            self.check_defined(ptr, size)?;
            self.check_relocations(ptr, size)?;
        } else {
            // We still don't want relocations on the *edges*
            self.check_relocation_edges(ptr, size)?;
        }

        let alloc = self.get(ptr.alloc_id)?;
        Extra::memory_accessed(&self.extra, ptr, size, MemoryAccess::Read)?;

        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&alloc.bytes[offset..offset + size.bytes() as usize])
    }

    #[inline]
    fn get_bytes(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align
    ) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes_internal(ptr, size, align, true)
    }

    /// It is the caller's responsibility to handle undefined and pointer bytes.
    /// However, this still checks that there are no relocations on the *edges*.
    #[inline]
    fn get_bytes_with_undef_and_ptr(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align
    ) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes_internal(ptr, size, align, false)
    }

    /// Just calling this already marks everything as defined and removes relocations,
    /// so be sure to actually put data there!
    fn get_bytes_mut(
        &mut self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_align(ptr.into(), align)?;
        self.check_bounds(ptr, size, true)?;

        self.mark_definedness(ptr, size, true)?;
        self.clear_relocations(ptr, size)?;

        let alloc = self.get_mut(ptr.alloc_id)?;
        Extra::memory_accessed(alloc, ptr, size, MemoryAccess::Write)?;

        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&mut alloc.bytes[offset..offset + size.bytes() as usize])
    }
}

/// Reading and writing
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    pub fn copy(
        &mut self,
        src: Scalar<Tag>,
        src_align: Align,
        dest: Scalar<Tag>,
        dest_align: Align,
        size: Size,
        nonoverlapping: bool,
    ) -> EvalResult<'tcx> {
        self.copy_repeatedly(src, src_align, dest, dest_align, size, 1, nonoverlapping)
    }

    pub fn copy_repeatedly(
        &mut self,
        src: Scalar<Tag>,
        src_align: Align,
        dest: Scalar<Tag>,
        dest_align: Align,
        size: Size,
        length: u64,
        nonoverlapping: bool,
    ) -> EvalResult<'tcx> {
        if size.bytes() == 0 {
            // Nothing to do for ZST, other than checking alignment and non-NULLness.
            self.check_align(src, src_align)?;
            self.check_align(dest, dest_align)?;
            return Ok(());
        }
        let src = src.to_ptr()?;
        let dest = dest.to_ptr()?;

        // first copy the relocations to a temporary buffer, because
        // `get_bytes_mut` will clear the relocations, which is correct,
        // since we don't want to keep any relocations at the target.
        // (`get_bytes_with_undef_and_ptr` below checks that there are no
        // relocations overlapping the edges; those would not be handled correctly).
        let relocations = {
            let relocations = self.relocations(src, size)?;
            let mut new_relocations = Vec::with_capacity(relocations.len() * (length as usize));
            for i in 0..length {
                new_relocations.extend(
                    relocations
                    .iter()
                    .map(|&(offset, reloc)| {
                    (offset + dest.offset - src.offset + (i * size * relocations.len() as u64),
                     reloc)
                    })
                );
            }

            new_relocations
        };

        // This also checks alignment, and relocation edges on the src.
        let src_bytes = self.get_bytes_with_undef_and_ptr(src, size, src_align)?.as_ptr();
        let dest_bytes = self.get_bytes_mut(dest, size * length, dest_align)?.as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        // The pointers above remain valid even if the `HashMap` table is moved around because they
        // point into the `Vec` storing the bytes.
        unsafe {
            assert_eq!(size.bytes() as usize as u64, size.bytes());
            if src.alloc_id == dest.alloc_id {
                if nonoverlapping {
                    if (src.offset <= dest.offset && src.offset + size > dest.offset) ||
                        (dest.offset <= src.offset && dest.offset + size > src.offset)
                    {
                        return err!(Intrinsic(
                            "copy_nonoverlapping called on overlapping ranges".to_string(),
                        ));
                    }
                }

                for i in 0..length {
                    ptr::copy(src_bytes,
                              dest_bytes.offset((size.bytes() * i) as isize),
                              size.bytes() as usize);
                }
            } else {
                for i in 0..length {
                    ptr::copy_nonoverlapping(src_bytes,
                                             dest_bytes.offset((size.bytes() * i) as isize),
                                             size.bytes() as usize);
                }
            }
        }

        // copy definedness to the destination
        self.copy_undef_mask(src, dest, size, length)?;
        // copy the relocations to the destination
        self.get_mut(dest.alloc_id)?.relocations.insert_presorted(relocations);

        Ok(())
    }

    pub fn read_c_str(&self, ptr: Pointer<Tag>) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        let offset = ptr.offset.bytes() as usize;
        match alloc.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                let p1 = Size::from_bytes((size + 1) as u64);
                self.check_relocations(ptr, p1)?;
                self.check_defined(ptr, p1)?;
                Ok(&alloc.bytes[offset..offset + size])
            }
            None => err!(UnterminatedCString(ptr.erase_tag())),
        }
    }

    pub fn check_bytes(
        &self,
        ptr: Scalar<Tag>,
        size: Size,
        allow_ptr_and_undef: bool,
    ) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        if size.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let ptr = ptr.to_ptr()?;
        // Check bounds, align and relocations on the edges
        self.get_bytes_with_undef_and_ptr(ptr, size, align)?;
        // Check undef and ptr
        if !allow_ptr_and_undef {
            self.check_defined(ptr, size)?;
            self.check_relocations(ptr, size)?;
        }
        Ok(())
    }

    pub fn read_bytes(&self, ptr: Scalar<Tag>, size: Size) -> EvalResult<'tcx, &[u8]> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        if size.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, align)
    }

    pub fn write_bytes(&mut self, ptr: Scalar<Tag>, src: &[u8]) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        if src.is_empty() {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, Size::from_bytes(src.len() as u64), align)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(
        &mut self,
        ptr: Scalar<Tag>,
        val: u8,
        count: Size
    ) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        if count.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, count, align)?;
        for b in bytes {
            *b = val;
        }
        Ok(())
    }

    /// Read a *non-ZST* scalar
    pub fn read_scalar(
        &self,
        ptr: Pointer<Tag>,
        ptr_align: Align,
        size: Size
    ) -> EvalResult<'tcx, ScalarMaybeUndef<Tag>> {
        // get_bytes_unchecked tests alignment and relocation edges
        let bytes = self.get_bytes_with_undef_and_ptr(
            ptr, size, ptr_align.min(self.int_align(size))
        )?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if self.check_defined(ptr, size).is_err() {
            // this inflates undefined bytes to the entire scalar, even if only a few
            // bytes are undefined
            return Ok(ScalarMaybeUndef::Undef);
        }
        // Now we do the actual reading
        let bits = read_target_uint(self.tcx.data_layout.endian, bytes).unwrap();
        // See if we got a pointer
        if size != self.pointer_size() {
            // *Now* better make sure that the inside also is free of relocations.
            self.check_relocations(ptr, size)?;
        } else {
            let alloc = self.get(ptr.alloc_id)?;
            match alloc.relocations.get(&ptr.offset) {
                Some(&(tag, alloc_id)) => {
                    let ptr = Pointer::new_with_tag(alloc_id, Size::from_bytes(bits as u64), tag);
                    return Ok(ScalarMaybeUndef::Scalar(ptr.into()))
                }
                None => {},
            }
        }
        // We don't. Just return the bits.
        Ok(ScalarMaybeUndef::Scalar(Scalar::from_uint(bits, size)))
    }

    pub fn read_ptr_sized(
        &self,
        ptr: Pointer<Tag>,
        ptr_align: Align
    ) -> EvalResult<'tcx, ScalarMaybeUndef<Tag>> {
        self.read_scalar(ptr, ptr_align, self.pointer_size())
    }

    /// Write a *non-ZST* scalar
    pub fn write_scalar(
        &mut self,
        ptr: Pointer<Tag>,
        ptr_align: Align,
        val: ScalarMaybeUndef<Tag>,
        type_size: Size,
    ) -> EvalResult<'tcx> {
        let val = match val {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => return self.mark_definedness(ptr, type_size, false),
        };

        let bytes = match val {
            Scalar::Ptr(val) => {
                assert_eq!(type_size, self.pointer_size());
                val.offset.bytes() as u128
            }

            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, type_size.bytes());
                debug_assert_eq!(truncate(bits, Size::from_bytes(size.into())), bits,
                    "Unexpected value of size {} when writing to memory", size);
                bits
            },
        };

        {
            // get_bytes_mut checks alignment
            let endian = self.tcx.data_layout.endian;
            let dst = self.get_bytes_mut(ptr, type_size, ptr_align)?;
            write_target_uint(endian, dst, bytes).unwrap();
        }

        // See if we have to also write a relocation
        match val {
            Scalar::Ptr(val) => {
                self.get_mut(ptr.alloc_id)?.relocations.insert(
                    ptr.offset,
                    (val.tag, val.alloc_id),
                );
            }
            _ => {}
        }

        Ok(())
    }

    pub fn write_ptr_sized(
        &mut self,
        ptr: Pointer<Tag>,
        ptr_align: Align,
        val: ScalarMaybeUndef<Tag>
    ) -> EvalResult<'tcx> {
        let ptr_size = self.pointer_size();
        self.write_scalar(ptr.into(), ptr_align, val, ptr_size)
    }

    fn int_align(&self, size: Size) -> Align {
        // We assume pointer-sized integers have the same alignment as pointers.
        // We also assume signed and unsigned integers of the same size have the same alignment.
        let ity = match size.bytes() {
            1 => layout::I8,
            2 => layout::I16,
            4 => layout::I32,
            8 => layout::I64,
            16 => layout::I128,
            _ => bug!("bad integer size: {}", size.bytes()),
        };
        ity.align(self)
    }
}

/// Relocations
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// Return all relocations overlapping with the given ptr-offset pair.
    fn relocations(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx, &[(Size, (Tag, AllocId))]> {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let start = ptr.offset.bytes().saturating_sub(self.pointer_size().bytes() - 1);
        let end = ptr.offset + size; // this does overflow checking
        Ok(self.get(ptr.alloc_id)?.relocations.range(Size::from_bytes(start)..end))
    }

    /// Check that there ar eno relocations overlapping with the given range.
    #[inline(always)]
    fn check_relocations(&self, ptr: Pointer<Tag>, size: Size) -> EvalResult<'tcx> {
        if self.relocations(ptr, size)?.len() != 0 {
            err!(ReadPointerAsBytes)
        } else {
            Ok(())
        }
    }

    /// Remove all relocations inside the given range.
    /// If there are relocations overlapping with the edges, they
    /// are removed as well *and* the bytes they cover are marked as
    /// uninitialized.  This is a somewhat odd "spooky action at a distance",
    /// but it allows strictly more code to run than if we would just error
    /// immediately in that case.
    fn clear_relocations(&mut self, ptr: Pointer<Tag>, size: Size) -> EvalResult<'tcx> {
        // Find the start and end of the given range and its outermost relocations.
        let (first, last) = {
            // Find all relocations overlapping the given range.
            let relocations = self.relocations(ptr, size)?;
            if relocations.is_empty() {
                return Ok(());
            }

            (relocations.first().unwrap().0,
             relocations.last().unwrap().0 + self.pointer_size())
        };
        let start = ptr.offset;
        let end = start + size;

        let alloc = self.get_mut(ptr.alloc_id)?;

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start {
            alloc.undef_mask.set_range(first, start, false);
        }
        if last > end {
            alloc.undef_mask.set_range(end, last, false);
        }

        // Forget all the relocations.
        alloc.relocations.remove_range(first..last);

        Ok(())
    }

    /// Error if there are relocations overlapping with the edges of the
    /// given memory range.
    #[inline]
    fn check_relocation_edges(&self, ptr: Pointer<Tag>, size: Size) -> EvalResult<'tcx> {
        self.check_relocations(ptr, Size::ZERO)?;
        self.check_relocations(ptr.offset(size, self)?, Size::ZERO)?;
        Ok(())
    }
}

/// Undefined bytes
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    // FIXME: Add a fast version for the common, nonoverlapping case
    fn copy_undef_mask(
        &mut self,
        src: Pointer<Tag>,
        dest: Pointer<Tag>,
        size: Size,
        repeat: u64,
    ) -> EvalResult<'tcx> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        assert_eq!(size.bytes() as usize as u64, size.bytes());

        let undef_mask = self.get(src.alloc_id)?.undef_mask.clone();
        let dest_allocation = self.get_mut(dest.alloc_id)?;

        for i in 0..size.bytes() {
            let defined = undef_mask.get(src.offset + Size::from_bytes(i));

            for j in 0..repeat {
                dest_allocation.undef_mask.set(
                    dest.offset + Size::from_bytes(i + (size.bytes() * j)),
                    defined
                );
            }
        }

        Ok(())
    }

    /// Checks that a range of bytes is defined. If not, returns the `ReadUndefBytes`
    /// error which will report the first byte which is undefined.
    #[inline]
    fn check_defined(&self, ptr: Pointer<Tag>, size: Size) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        alloc.undef_mask.is_range_defined(
            ptr.offset,
            ptr.offset + size,
        ).or_else(|idx| err!(ReadUndefBytes(idx)))
    }

    pub fn mark_definedness(
        &mut self,
        ptr: Pointer<Tag>,
        size: Size,
        new_state: bool,
    ) -> EvalResult<'tcx> {
        if size.bytes() == 0 {
            return Ok(());
        }
        let alloc = self.get_mut(ptr.alloc_id)?;
        alloc.undef_mask.set_range(
            ptr.offset,
            ptr.offset + size,
            new_state,
        );
        Ok(())
    }
}

impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// Check that the pointer is aligned AND non-NULL. This supports ZSTs in two ways:
    /// You can pass a scalar, and a `Pointer` does not have to actually still be allocated.
    pub fn check_align(
        &self,
        ptr: Scalar<Tag>,
        required_align: Align
    ) -> EvalResult<'tcx> {
        // Check non-NULL/Undef, extract offset
        let (offset, alloc_align) = match ptr {
            Scalar::Ptr(ptr) => {
                let (size, align) = self.get_size_and_align(ptr.alloc_id);
                // check this is not NULL -- which we can ensure only if this is in-bounds
                // of some (potentially dead) allocation.
                if ptr.offset > size {
                    return err!(PointerOutOfBounds {
                        ptr: ptr.erase_tag(),
                        access: true,
                        allocation_size: size,
                    });
                };
                // keep data for alignment check
                (ptr.offset.bytes(), align)
            }
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, self.pointer_size().bytes());
                assert!(bits < (1u128 << self.pointer_size().bits()));
                // check this is not NULL
                if bits == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                // the "base address" is 0 and hence always aligned
                (bits as u64, required_align)
            }
        };
        // Check alignment
        if alloc_align.abi() < required_align.abi() {
            return err!(AlignmentCheckFailed {
                has: alloc_align,
                required: required_align,
            });
        }
        if offset % required_align.abi() == 0 {
            Ok(())
        } else {
            let has = offset % required_align.abi();
            err!(AlignmentCheckFailed {
                has: Align::from_bytes(has, has).unwrap(),
                required: required_align,
            })
        }
    }

    /// Check if the pointer is "in-bounds". Notice that a pointer pointing at the end
    /// of an allocation (i.e., at the first *inaccessible* location) *is* considered
    /// in-bounds!  This follows C's/LLVM's rules.  The `access` boolean is just used
    /// for the error message.
    /// If you want to check bounds before doing a memory access, be sure to
    /// check the pointer one past the end of your access, then everything will
    /// work out exactly.
    pub fn check_bounds_ptr(&self, ptr: Pointer<Tag>, access: bool) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        let allocation_size = alloc.bytes.len() as u64;
        if ptr.offset.bytes() > allocation_size {
            return err!(PointerOutOfBounds {
                ptr: ptr.erase_tag(),
                access,
                allocation_size: Size::from_bytes(allocation_size),
            });
        }
        Ok(())
    }

    /// Check if the memory range beginning at `ptr` and of size `Size` is "in-bounds".
    #[inline(always)]
    pub fn check_bounds(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        access: bool
    ) -> EvalResult<'tcx> {
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds_ptr(ptr.offset(size, &*self)?, access)
    }
}