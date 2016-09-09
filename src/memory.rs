use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian, BigEndian, self};
use std::collections::Bound::{Included, Excluded};
use std::collections::{btree_map, BTreeMap, HashMap, HashSet, VecDeque};
use std::{fmt, iter, ptr};

use rustc::hir::def_id::DefId;
use rustc::ty::{BareFnTy, ClosureTy, ClosureSubsts};
use rustc::ty::subst::Substs;
use rustc::ty::layout::{self, TargetDataLayout};

use error::{EvalError, EvalResult};
use primval::PrimVal;

////////////////////////////////////////////////////////////////////////////////
// Allocations and pointers
////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct AllocId(pub u64);

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct Allocation {
    pub bytes: Vec<u8>,
    pub relocations: BTreeMap<usize, AllocId>,
    pub undef_mask: UndefMask,
    pub align: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: usize,
}

impl Pointer {
    pub fn offset(self, i: isize) -> Self {
        Pointer { offset: (self.offset as isize + i) as usize, ..self }
    }
    pub fn points_to_zst(&self) -> bool {
        self.alloc_id == ZST_ALLOC_ID
    }
    fn zst_ptr() -> Self {
        Pointer {
            alloc_id: ZST_ALLOC_ID,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct FunctionDefinition<'tcx> {
    pub def_id: DefId,
    pub kind: FunctionKind<'tcx>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum FunctionKind<'tcx> {
    Closure {
        substs: ClosureSubsts<'tcx>,
        ty: ClosureTy<'tcx>,
    },
    Function {
        substs: &'tcx Substs<'tcx>,
        ty: &'tcx BareFnTy<'tcx>,
    }
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

pub struct Memory<'a, 'tcx> {
    /// Actual memory allocations (arbitrary bytes, may contain pointers into other allocations)
    alloc_map: HashMap<AllocId, Allocation>,
    /// Number of virtual bytes allocated
    memory_usage: usize,
    /// Maximum number of virtual bytes that may be allocated
    memory_size: usize,
    /// Function "allocations". They exist solely so pointers have something to point to, and
    /// we can figure out what they point to.
    functions: HashMap<AllocId, FunctionDefinition<'tcx>>,
    /// Inverse map of `functions` so we don't allocate a new pointer every time we need one
    function_alloc_cache: HashMap<FunctionDefinition<'tcx>, AllocId>,
    next_id: AllocId,
    pub layout: &'a TargetDataLayout,
}

const ZST_ALLOC_ID: AllocId = AllocId(0);

impl<'a, 'tcx> Memory<'a, 'tcx> {
    pub fn new(layout: &'a TargetDataLayout, max_memory: usize) -> Self {
        let mut mem = Memory {
            alloc_map: HashMap::new(),
            functions: HashMap::new(),
            function_alloc_cache: HashMap::new(),
            next_id: AllocId(1),
            layout: layout,
            memory_size: max_memory,
            memory_usage: 0,
        };
        // alloc id 0 is reserved for ZSTs, this is an optimization to prevent ZST
        // (e.g. function items, (), [], ...) from requiring memory
        let alloc = Allocation {
            bytes: Vec::new(),
            relocations: BTreeMap::new(),
            undef_mask: UndefMask::new(0),
            align: 1,
        };
        mem.alloc_map.insert(ZST_ALLOC_ID, alloc);
        // check that additional zst allocs work
        debug_assert!(mem.allocate(0, 1).unwrap().points_to_zst());
        debug_assert!(mem.get(ZST_ALLOC_ID).is_ok());
        mem
    }

    pub fn allocations(&self) -> ::std::collections::hash_map::Iter<AllocId, Allocation> {
        self.alloc_map.iter()
    }

    pub fn create_closure_ptr(&mut self, def_id: DefId, substs: ClosureSubsts<'tcx>, fn_ty: ClosureTy<'tcx>) -> Pointer {
        self.create_fn_alloc(FunctionDefinition {
            def_id: def_id,
            kind: FunctionKind::Closure {
                substs: substs,
                ty: fn_ty,
            }
        })
    }

    pub fn create_fn_ptr(&mut self, def_id: DefId, substs: &'tcx Substs<'tcx>, fn_ty: &'tcx BareFnTy<'tcx>) -> Pointer {
        self.create_fn_alloc(FunctionDefinition {
            def_id: def_id,
            kind: FunctionKind::Function {
                substs: substs,
                ty: fn_ty,
            }
        })
    }

    fn create_fn_alloc(&mut self, def: FunctionDefinition<'tcx>) -> Pointer {
        if let Some(&alloc_id) = self.function_alloc_cache.get(&def) {
            return Pointer {
                alloc_id: alloc_id,
                offset: 0,
            };
        }
        let id = self.next_id;
        debug!("creating fn ptr: {}", id);
        self.next_id.0 += 1;
        self.functions.insert(id, def.clone());
        self.function_alloc_cache.insert(def, id);
        Pointer {
            alloc_id: id,
            offset: 0,
        }
    }

    pub fn allocate(&mut self, size: usize, align: usize) -> EvalResult<'tcx, Pointer> {
        assert!(align != 0);
        if size == 0 {
            return Ok(Pointer::zst_ptr());
        }
        if self.memory_size - self.memory_usage < size {
            return Err(EvalError::OutOfMemory {
                allocation_size: size,
                memory_size: self.memory_size,
                memory_usage: self.memory_usage,
            });
        }
        self.memory_usage += size;
        let alloc = Allocation {
            bytes: vec![0; size],
            relocations: BTreeMap::new(),
            undef_mask: UndefMask::new(size),
            align: align,
        };
        let id = self.next_id;
        self.next_id.0 += 1;
        self.alloc_map.insert(id, alloc);
        Ok(Pointer {
            alloc_id: id,
            offset: 0,
        })
    }

    // TODO(solson): Track which allocations were returned from __rust_allocate and report an error
    // when reallocating/deallocating any others.
    pub fn reallocate(&mut self, ptr: Pointer, new_size: usize, align: usize) -> EvalResult<'tcx, Pointer> {
        // TODO(solson): Report error about non-__rust_allocate'd pointer.
        if ptr.offset != 0 {
            return Err(EvalError::Unimplemented(format!("bad pointer offset: {}", ptr.offset)));
        }
        if ptr.points_to_zst() {
            return self.allocate(new_size, align);
        }

        let size = self.get(ptr.alloc_id)?.bytes.len();

        if new_size > size {
            let amount = new_size - size;
            self.memory_usage += amount;
            let alloc = self.get_mut(ptr.alloc_id)?;
            alloc.bytes.extend(iter::repeat(0).take(amount));
            alloc.undef_mask.grow(amount, false);
        } else if size > new_size {
            self.memory_usage -= size - new_size;
            self.clear_relocations(ptr.offset(new_size as isize), size - new_size)?;
            let alloc = self.get_mut(ptr.alloc_id)?;
            alloc.bytes.truncate(new_size);
            alloc.bytes.shrink_to_fit();
            alloc.undef_mask.truncate(new_size);
        }

        Ok(Pointer {
            alloc_id: ptr.alloc_id,
            offset: 0,
        })
    }

    // TODO(solson): See comment on `reallocate`.
    pub fn deallocate(&mut self, ptr: Pointer) -> EvalResult<'tcx, ()> {
        if ptr.points_to_zst() {
            return Ok(());
        }
        if ptr.offset != 0 {
            // TODO(solson): Report error about non-__rust_allocate'd pointer.
            return Err(EvalError::Unimplemented(format!("bad pointer offset: {}", ptr.offset)));
        }

        if let Some(alloc) = self.alloc_map.remove(&ptr.alloc_id) {
            self.memory_usage -= alloc.bytes.len();
        } else {
            debug!("deallocated a pointer twice: {}", ptr.alloc_id);
            // TODO(solson): Report error about erroneous free. This is blocked on properly tracking
            // already-dropped state since this if-statement is entered even in safe code without
            // it.
        }
        debug!("deallocated : {}", ptr.alloc_id);

        Ok(())
    }

    pub fn pointer_size(&self) -> usize {
        self.layout.pointer_size.bytes() as usize
    }

    pub fn endianess(&self) -> layout::Endian {
        self.layout.endian
    }

    pub fn check_align(&self, ptr: Pointer, align: usize) -> EvalResult<'tcx, ()> {
        let alloc = self.get(ptr.alloc_id)?;
        if alloc.align < align {
            return Err(EvalError::AlignmentCheckFailed {
                has: alloc.align,
                required: align,
            });
        }
        if ptr.offset % align == 0 {
            Ok(())
        } else {
            Err(EvalError::AlignmentCheckFailed {
                has: ptr.offset % align,
                required: align,
            })
        }
    }
}

/// Allocation accessors
impl<'a, 'tcx> Memory<'a, 'tcx> {
    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation> {
        match self.alloc_map.get(&id) {
            Some(alloc) => Ok(alloc),
            None => match self.functions.get(&id) {
                Some(_) => Err(EvalError::DerefFunctionPointer),
                None => Err(EvalError::DanglingPointerDeref),
            }
        }
    }

    pub fn get_mut(&mut self, id: AllocId) -> EvalResult<'tcx, &mut Allocation> {
        match self.alloc_map.get_mut(&id) {
            Some(alloc) => Ok(alloc),
            None => match self.functions.get(&id) {
                Some(_) => Err(EvalError::DerefFunctionPointer),
                None => Err(EvalError::DanglingPointerDeref),
            }
        }
    }

    pub fn get_closure(&self, id: AllocId) -> EvalResult<'tcx, (DefId, ClosureSubsts<'tcx>, ClosureTy<'tcx>)> {
        debug!("reading closure fn ptr: {}", id);
        match self.functions.get(&id) {
            Some(&FunctionDefinition {
                def_id,
                kind: FunctionKind::Closure { ref substs, ref ty }
            }) => Ok((def_id, substs.clone(), ty.clone())),
            Some(&FunctionDefinition {
                kind: FunctionKind::Function { .. }, ..
            }) => Err(EvalError::CalledClosureAsFunction),
            None => match self.alloc_map.get(&id) {
                Some(_) => Err(EvalError::ExecuteMemory),
                None => Err(EvalError::InvalidFunctionPointer),
            }
        }
    }

    pub fn get_fn(&self, id: AllocId) -> EvalResult<'tcx, (DefId, &'tcx Substs<'tcx>, &'tcx BareFnTy<'tcx>)> {
        debug!("reading fn ptr: {}", id);
        match self.functions.get(&id) {
            Some(&FunctionDefinition {
                def_id,
                kind: FunctionKind::Function { substs, ty }
            }) => Ok((def_id, substs, ty)),
            Some(&FunctionDefinition {
                kind: FunctionKind::Closure { .. }, ..
            }) => Err(EvalError::CalledClosureAsFunction),
            None => match self.alloc_map.get(&id) {
                Some(_) => Err(EvalError::ExecuteMemory),
                None => Err(EvalError::InvalidFunctionPointer),
            }
        }
    }

    /// Print an allocation and all allocations it points to, recursively.
    pub fn dump(&self, id: AllocId) {
        let mut allocs_seen = HashSet::new();
        let mut allocs_to_print = VecDeque::new();
        allocs_to_print.push_back(id);

        while let Some(id) = allocs_to_print.pop_front() {
            allocs_seen.insert(id);
            let prefix = format!("Alloc {:<5} ", format!("{}:", id));
            print!("{}", prefix);
            let mut relocations = vec![];

            let alloc = match (self.alloc_map.get(&id), self.functions.get(&id)) {
                (Some(a), None) => a,
                (None, Some(_)) => {
                    // FIXME: print function name
                    println!("function pointer");
                    continue;
                },
                (None, None) => {
                    println!("(deallocated)");
                    continue;
                },
                (Some(_), Some(_)) => bug!("miri invariant broken: an allocation id exists that points to both a function and a memory location"),
            };

            for i in 0..alloc.bytes.len() {
                if let Some(&target_id) = alloc.relocations.get(&i) {
                    if !allocs_seen.contains(&target_id) {
                        allocs_to_print.push_back(target_id);
                    }
                    relocations.push((i, target_id));
                }
                if alloc.undef_mask.is_range_defined(i, i + 1) {
                    print!("{:02x} ", alloc.bytes[i]);
                } else {
                    print!("__ ");
                }
            }
            println!("({} bytes)", alloc.bytes.len());

            if !relocations.is_empty() {
                print!("{:1$}", "", prefix.len()); // Print spaces.
                let mut pos = 0;
                let relocation_width = (self.pointer_size() - 1) * 3;
                for (i, target_id) in relocations {
                    print!("{:1$}", "", (i - pos) * 3);
                    print!("└{0:─^1$}┘ ", format!("({})", target_id), relocation_width);
                    pos = i + self.pointer_size();
                }
                println!("");
            }
        }
    }
}

/// Byte accessors
impl<'a, 'tcx> Memory<'a, 'tcx> {
    fn get_bytes_unchecked(&self, ptr: Pointer, size: usize) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        if ptr.offset + size > alloc.bytes.len() {
            return Err(EvalError::PointerOutOfBounds {
                ptr: ptr,
                size: size,
                allocation_size: alloc.bytes.len(),
            });
        }
        Ok(&alloc.bytes[ptr.offset..ptr.offset + size])
    }

    fn get_bytes_unchecked_mut(&mut self, ptr: Pointer, size: usize) -> EvalResult<'tcx, &mut [u8]> {
        let alloc = self.get_mut(ptr.alloc_id)?;
        if ptr.offset + size > alloc.bytes.len() {
            return Err(EvalError::PointerOutOfBounds {
                ptr: ptr,
                size: size,
                allocation_size: alloc.bytes.len(),
            });
        }
        Ok(&mut alloc.bytes[ptr.offset..ptr.offset + size])
    }

    fn get_bytes(&self, ptr: Pointer, size: usize, align: usize) -> EvalResult<'tcx, &[u8]> {
        self.check_align(ptr, align)?;
        if self.relocations(ptr, size)?.count() != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        self.check_defined(ptr, size)?;
        self.get_bytes_unchecked(ptr, size)
    }

    fn get_bytes_mut(&mut self, ptr: Pointer, size: usize, align: usize) -> EvalResult<'tcx, &mut [u8]> {
        self.check_align(ptr, align)?;
        self.clear_relocations(ptr, size)?;
        self.mark_definedness(ptr, size, true)?;
        self.get_bytes_unchecked_mut(ptr, size)
    }
}

/// Reading and writing
impl<'a, 'tcx> Memory<'a, 'tcx> {
    pub fn copy(&mut self, src: Pointer, dest: Pointer, size: usize, align: usize) -> EvalResult<'tcx, ()> {
        self.check_relocation_edges(src, size)?;

        let src_bytes = self.get_bytes_unchecked(src, size)?.as_ptr();
        let dest_bytes = self.get_bytes_mut(dest, size, align)?.as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        unsafe {
            if src.alloc_id == dest.alloc_id {
                ptr::copy(src_bytes, dest_bytes, size);
            } else {
                ptr::copy_nonoverlapping(src_bytes, dest_bytes, size);
            }
        }

        self.copy_undef_mask(src, dest, size)?;
        self.copy_relocations(src, dest, size)?;

        Ok(())
    }

    pub fn read_bytes(&self, ptr: Pointer, size: usize) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes(ptr, size, 1)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<'tcx, ()> {
        let bytes = self.get_bytes_mut(ptr, src.len(), 1)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(&mut self, ptr: Pointer, val: u8, count: usize) -> EvalResult<'tcx, ()> {
        let bytes = self.get_bytes_mut(ptr, count, 1)?;
        for b in bytes { *b = val; }
        Ok(())
    }

    pub fn read_ptr(&self, ptr: Pointer) -> EvalResult<'tcx, Pointer> {
        let size = self.pointer_size();
        self.check_defined(ptr, size)?;
        let endianess = self.endianess();
        let bytes = self.get_bytes_unchecked(ptr, size)?;
        let offset = read_target_uint(endianess, bytes).unwrap() as usize;
        let alloc = self.get(ptr.alloc_id)?;
        match alloc.relocations.get(&ptr.offset) {
            Some(&alloc_id) => Ok(Pointer { alloc_id: alloc_id, offset: offset }),
            None => Err(EvalError::ReadBytesAsPointer),
        }
    }

    pub fn write_ptr(&mut self, dest: Pointer, ptr: Pointer) -> EvalResult<'tcx, ()> {
        self.write_usize(dest, ptr.offset as u64)?;
        self.get_mut(dest.alloc_id)?.relocations.insert(dest.offset, ptr.alloc_id);
        Ok(())
    }

    pub fn write_primval(&mut self, ptr: Pointer, val: PrimVal) -> EvalResult<'tcx, ()> {
        let pointer_size = self.pointer_size();
        match val {
            PrimVal::Bool(b) => self.write_bool(ptr, b),
            PrimVal::I8(n)   => self.write_int(ptr, n as i64, 1),
            PrimVal::I16(n)  => self.write_int(ptr, n as i64, 2),
            PrimVal::I32(n)  => self.write_int(ptr, n as i64, 4),
            PrimVal::I64(n)  => self.write_int(ptr, n as i64, 8),
            PrimVal::U8(n)   => self.write_uint(ptr, n as u64, 1),
            PrimVal::U16(n)  => self.write_uint(ptr, n as u64, 2),
            PrimVal::U32(n)  => self.write_uint(ptr, n as u64, 4),
            PrimVal::U64(n)  => self.write_uint(ptr, n as u64, 8),
            PrimVal::Char(c) => self.write_uint(ptr, c as u64, 4),
            PrimVal::IntegerPtr(n) => self.write_uint(ptr, n as u64, pointer_size),
            PrimVal::F32(f) => self.write_f32(ptr, f),
            PrimVal::F64(f) => self.write_f64(ptr, f),
            PrimVal::FnPtr(p) |
            PrimVal::AbstractPtr(p) => self.write_ptr(ptr, p),
        }
    }

    pub fn read_bool(&self, ptr: Pointer) -> EvalResult<'tcx, bool> {
        let bytes = self.get_bytes(ptr, 1, self.layout.i1_align.abi() as usize)?;
        match bytes[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(EvalError::InvalidBool),
        }
    }

    pub fn write_bool(&mut self, ptr: Pointer, b: bool) -> EvalResult<'tcx, ()> {
        let align = self.layout.i1_align.abi() as usize;
        self.get_bytes_mut(ptr, 1, align)
            .map(|bytes| bytes[0] = b as u8)
    }

    fn int_align(&self, size: usize) -> EvalResult<'tcx, usize> {
        match size {
            1 => Ok(self.layout.i8_align.abi() as usize),
            2 => Ok(self.layout.i16_align.abi() as usize),
            4 => Ok(self.layout.i32_align.abi() as usize),
            8 => Ok(self.layout.i64_align.abi() as usize),
            _ => bug!("bad integer size"),
        }
    }

    pub fn read_int(&self, ptr: Pointer, size: usize) -> EvalResult<'tcx, i64> {
        let align = self.int_align(size)?;
        self.get_bytes(ptr, size, align).map(|b| read_target_int(self.endianess(), b).unwrap())
    }

    pub fn write_int(&mut self, ptr: Pointer, n: i64, size: usize) -> EvalResult<'tcx, ()> {
        let align = self.int_align(size)?;
        let endianess = self.endianess();
        let b = self.get_bytes_mut(ptr, size, align)?;
        write_target_int(endianess, b, n).unwrap();
        Ok(())
    }

    pub fn read_uint(&self, ptr: Pointer, size: usize) -> EvalResult<'tcx, u64> {
        let align = self.int_align(size)?;
        self.get_bytes(ptr, size, align).map(|b| read_target_uint(self.endianess(), b).unwrap())
    }

    pub fn write_uint(&mut self, ptr: Pointer, n: u64, size: usize) -> EvalResult<'tcx, ()> {
        let align = self.int_align(size)?;
        let endianess = self.endianess();
        let b = self.get_bytes_mut(ptr, size, align)?;
        write_target_uint(endianess, b, n).unwrap();
        Ok(())
    }

    pub fn read_isize(&self, ptr: Pointer) -> EvalResult<'tcx, i64> {
        self.read_int(ptr, self.pointer_size())
    }

    pub fn write_isize(&mut self, ptr: Pointer, n: i64) -> EvalResult<'tcx, ()> {
        let size = self.pointer_size();
        self.write_int(ptr, n, size)
    }

    pub fn read_usize(&self, ptr: Pointer) -> EvalResult<'tcx, u64> {
        self.read_uint(ptr, self.pointer_size())
    }

    pub fn write_usize(&mut self, ptr: Pointer, n: u64) -> EvalResult<'tcx, ()> {
        let size = self.pointer_size();
        self.write_uint(ptr, n, size)
    }

    pub fn write_f32(&mut self, ptr: Pointer, f: f32) -> EvalResult<'tcx, ()> {
        let endianess = self.endianess();
        let align = self.layout.f32_align.abi() as usize;
        let b = self.get_bytes_mut(ptr, 4, align)?;
        write_target_f32(endianess, b, f).unwrap();
        Ok(())
    }

    pub fn write_f64(&mut self, ptr: Pointer, f: f64) -> EvalResult<'tcx, ()> {
        let endianess = self.endianess();
        let align = self.layout.f64_align.abi() as usize;
        let b = self.get_bytes_mut(ptr, 8, align)?;
        write_target_f64(endianess, b, f).unwrap();
        Ok(())
    }

    pub fn read_f32(&self, ptr: Pointer) -> EvalResult<'tcx, f32> {
        self.get_bytes(ptr, 4, self.layout.f32_align.abi() as usize)
            .map(|b| read_target_f32(self.endianess(), b).unwrap())
    }

    pub fn read_f64(&self, ptr: Pointer) -> EvalResult<'tcx, f64> {
        self.get_bytes(ptr, 8, self.layout.f64_align.abi() as usize)
            .map(|b| read_target_f64(self.endianess(), b).unwrap())
    }
}

/// Relocations
impl<'a, 'tcx> Memory<'a, 'tcx> {
    fn relocations(&self, ptr: Pointer, size: usize)
        -> EvalResult<'tcx, btree_map::Range<usize, AllocId>>
    {
        let start = ptr.offset.saturating_sub(self.pointer_size() - 1);
        let end = ptr.offset + size;
        Ok(self.get(ptr.alloc_id)?.relocations.range(Included(&start), Excluded(&end)))
    }

    fn clear_relocations(&mut self, ptr: Pointer, size: usize) -> EvalResult<'tcx, ()> {
        // Find all relocations overlapping the given range.
        let keys: Vec<_> = self.relocations(ptr, size)?.map(|(&k, _)| k).collect();
        if keys.is_empty() { return Ok(()); }

        // Find the start and end of the given range and its outermost relocations.
        let start = ptr.offset;
        let end = start + size;
        let first = *keys.first().unwrap();
        let last = *keys.last().unwrap() + self.pointer_size();

        let alloc = self.get_mut(ptr.alloc_id)?;

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start { alloc.undef_mask.set_range(first, start, false); }
        if last > end { alloc.undef_mask.set_range(end, last, false); }

        // Forget all the relocations.
        for k in keys { alloc.relocations.remove(&k); }

        Ok(())
    }

    fn check_relocation_edges(&self, ptr: Pointer, size: usize) -> EvalResult<'tcx, ()> {
        let overlapping_start = self.relocations(ptr, 0)?.count();
        let overlapping_end = self.relocations(ptr.offset(size as isize), 0)?.count();
        if overlapping_start + overlapping_end != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        Ok(())
    }

    fn copy_relocations(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<'tcx, ()> {
        let relocations: Vec<_> = self.relocations(src, size)?
            .map(|(&offset, &alloc_id)| {
                // Update relocation offsets for the new positions in the destination allocation.
                (offset + dest.offset - src.offset, alloc_id)
            })
            .collect();
        self.get_mut(dest.alloc_id)?.relocations.extend(relocations);
        Ok(())
    }
}

/// Undefined bytes
impl<'a, 'tcx> Memory<'a, 'tcx> {
    // FIXME(solson): This is a very naive, slow version.
    fn copy_undef_mask(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<'tcx, ()> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        let mut v = Vec::with_capacity(size);
        for i in 0..size {
            let defined = self.get(src.alloc_id)?.undef_mask.get(src.offset + i);
            v.push(defined);
        }
        for (i, defined) in v.into_iter().enumerate() {
            self.get_mut(dest.alloc_id)?.undef_mask.set(dest.offset + i, defined);
        }
        Ok(())
    }

    fn check_defined(&self, ptr: Pointer, size: usize) -> EvalResult<'tcx, ()> {
        let alloc = self.get(ptr.alloc_id)?;
        if !alloc.undef_mask.is_range_defined(ptr.offset, ptr.offset + size) {
            return Err(EvalError::ReadUndefBytes);
        }
        Ok(())
    }

    pub fn mark_definedness(&mut self, ptr: Pointer, size: usize, new_state: bool)
        -> EvalResult<'tcx, ()>
    {
        let mut alloc = self.get_mut(ptr.alloc_id)?;
        alloc.undef_mask.set_range(ptr.offset, ptr.offset + size, new_state);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to access integers in the target endianess
////////////////////////////////////////////////////////////////////////////////

fn write_target_uint(endianess: layout::Endian, mut target: &mut [u8], data: u64) -> Result<(), byteorder::Error> {
    let len = target.len();
    match endianess {
        layout::Endian::Little => target.write_uint::<LittleEndian>(data, len),
        layout::Endian::Big => target.write_uint::<BigEndian>(data, len),
    }
}
fn write_target_int(endianess: layout::Endian, mut target: &mut [u8], data: i64) -> Result<(), byteorder::Error> {
    let len = target.len();
    match endianess {
        layout::Endian::Little => target.write_int::<LittleEndian>(data, len),
        layout::Endian::Big => target.write_int::<BigEndian>(data, len),
    }
}

fn read_target_uint(endianess: layout::Endian, mut source: &[u8]) -> Result<u64, byteorder::Error> {
    match endianess {
        layout::Endian::Little => source.read_uint::<LittleEndian>(source.len()),
        layout::Endian::Big => source.read_uint::<BigEndian>(source.len()),
    }
}
fn read_target_int(endianess: layout::Endian, mut source: &[u8]) -> Result<i64, byteorder::Error> {
    match endianess {
        layout::Endian::Little => source.read_int::<LittleEndian>(source.len()),
        layout::Endian::Big => source.read_int::<BigEndian>(source.len()),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to access floats in the target endianess
////////////////////////////////////////////////////////////////////////////////

fn write_target_f32(endianess: layout::Endian, mut target: &mut [u8], data: f32) -> Result<(), byteorder::Error> {
    match endianess {
        layout::Endian::Little => target.write_f32::<LittleEndian>(data),
        layout::Endian::Big => target.write_f32::<BigEndian>(data),
    }
}
fn write_target_f64(endianess: layout::Endian, mut target: &mut [u8], data: f64) -> Result<(), byteorder::Error> {
    match endianess {
        layout::Endian::Little => target.write_f64::<LittleEndian>(data),
        layout::Endian::Big => target.write_f64::<BigEndian>(data),
    }
}

fn read_target_f32(endianess: layout::Endian, mut source: &[u8]) -> Result<f32, byteorder::Error> {
    match endianess {
        layout::Endian::Little => source.read_f32::<LittleEndian>(),
        layout::Endian::Big => source.read_f32::<BigEndian>(),
    }
}
fn read_target_f64(endianess: layout::Endian, mut source: &[u8]) -> Result<f64, byteorder::Error> {
    match endianess {
        layout::Endian::Little => source.read_f64::<LittleEndian>(),
        layout::Endian::Big => source.read_f64::<BigEndian>(),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;
const BLOCK_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: usize,
}

impl UndefMask {
    fn new(size: usize) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: 0,
        };
        m.grow(size, false);
        m
    }

    /// Check whether the range `start..end` (end-exclusive) is entirely defined.
    pub fn is_range_defined(&self, start: usize, end: usize) -> bool {
        if end > self.len { return false; }
        for i in start..end {
            if !self.get(i) { return false; }
        }
        true
    }

    fn set_range(&mut self, start: usize, end: usize, new_state: bool) {
        let len = self.len;
        if end > len { self.grow(end - len, new_state); }
        self.set_range_inbounds(start, end, new_state);
    }

    fn set_range_inbounds(&mut self, start: usize, end: usize, new_state: bool) {
        for i in start..end { self.set(i, new_state); }
    }

    fn get(&self, i: usize) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & 1 << bit) != 0
    }

    fn set(&mut self, i: usize, new_state: bool) {
        let (block, bit) = bit_index(i);
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    fn grow(&mut self, amount: usize, new_state: bool) {
        let unused_trailing_bits = self.blocks.len() * BLOCK_SIZE - self.len;
        if amount > unused_trailing_bits {
            let additional_blocks = amount / BLOCK_SIZE + 1;
            self.blocks.extend(iter::repeat(0).take(additional_blocks));
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state);
    }

    fn truncate(&mut self, length: usize) {
        self.len = length;
        self.blocks.truncate(self.len / BLOCK_SIZE + 1);
        self.blocks.shrink_to_fit();
    }
}

fn bit_index(bits: usize) -> (usize, usize) {
    (bits / BLOCK_SIZE, bits % BLOCK_SIZE)
}
