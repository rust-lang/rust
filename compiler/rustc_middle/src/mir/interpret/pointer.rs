use std::fmt;
use std::num::NonZero;

use rustc_abi::{HasDataLayout, Size};
use rustc_data_structures::static_assert_size;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};

use super::AllocId;

////////////////////////////////////////////////////////////////////////////////
// Pointer arithmetic
////////////////////////////////////////////////////////////////////////////////

pub trait PointerArithmetic: HasDataLayout {
    // These are not supposed to be overridden.

    #[inline(always)]
    fn pointer_size(&self) -> Size {
        self.data_layout().pointer_size()
    }

    #[inline(always)]
    fn max_size_of_val(&self) -> Size {
        Size::from_bytes(self.target_isize_max())
    }

    #[inline]
    fn target_usize_max(&self) -> u64 {
        self.pointer_size().unsigned_int_max().try_into().unwrap()
    }

    #[inline]
    fn target_isize_min(&self) -> i64 {
        self.pointer_size().signed_int_min().try_into().unwrap()
    }

    #[inline]
    fn target_isize_max(&self) -> i64 {
        self.pointer_size().signed_int_max().try_into().unwrap()
    }

    #[inline]
    fn truncate_to_target_usize(&self, val: u64) -> u64 {
        self.pointer_size().truncate(val.into()).try_into().unwrap()
    }

    #[inline]
    fn sign_extend_to_target_isize(&self, val: u64) -> i64 {
        self.pointer_size().sign_extend(val.into()).try_into().unwrap()
    }
}

impl<T: HasDataLayout> PointerArithmetic for T {}

/// This trait abstracts over the kind of provenance that is associated with a `Pointer`. It is
/// mostly opaque; the `Machine` trait extends it with some more operations that also have access to
/// some global state.
/// The `Debug` rendering is used to display bare provenance, and for the default impl of `fmt`.
pub trait Provenance: Copy + PartialEq + fmt::Debug + 'static {
    /// Says whether the `offset` field of `Pointer`s with this provenance is the actual physical address.
    /// - If `false`, the offset *must* be relative. This means the bytes representing a pointer are
    ///   different from what the Abstract Machine prescribes, so the interpreter must prevent any
    ///   operation that would inspect the underlying bytes of a pointer, such as ptr-to-int
    ///   transmutation. A `ReadPointerAsBytes` error will be raised in such situations.
    /// - If `true`, the interpreter will permit operations to inspect the underlying bytes of a
    ///   pointer, and implement ptr-to-int transmutation by stripping provenance.
    const OFFSET_IS_ADDR: bool;

    /// If wildcard provenance is implemented, contains the unique, general wildcard provenance variant.
    const WILDCARD: Option<Self>;

    /// Determines how a pointer should be printed.
    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// If `OFFSET_IS_ADDR == false`, provenance must always be able to
    /// identify the allocation this ptr points to (i.e., this must return `Some`).
    /// Otherwise this function is best-effort (but must agree with `Machine::ptr_get_alloc`).
    /// (Identifying the offset in that allocation, however, is harder -- use `Memory::ptr_get_alloc` for that.)
    fn get_alloc_id(self) -> Option<AllocId>;

    /// Defines the 'join' of provenance: what happens when doing a pointer load and different bytes have different provenance.
    fn join(left: Self, right: Self) -> Option<Self>;
}

/// The type of provenance in the compile-time interpreter.
/// This is a packed representation of:
/// - an `AllocId` (non-zero)
/// - an `immutable: bool`
/// - a `shared_ref: bool`
///
/// with the extra invariant that if `immutable` is `true`, then so
/// is `shared_ref`.
#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CtfeProvenance(NonZero<u64>);

impl From<AllocId> for CtfeProvenance {
    fn from(value: AllocId) -> Self {
        let prov = CtfeProvenance(value.0);
        assert!(
            prov.alloc_id() == value,
            "`AllocId` with the highest bits set cannot be used in CTFE"
        );
        prov
    }
}

impl fmt::Debug for CtfeProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.alloc_id(), f)?; // propagates `alternate` flag
        if self.immutable() {
            write!(f, "<imm>")?;
        }
        Ok(())
    }
}

const IMMUTABLE_MASK: u64 = 1 << 63; // the highest bit
const SHARED_REF_MASK: u64 = 1 << 62;
const ALLOC_ID_MASK: u64 = u64::MAX & !IMMUTABLE_MASK & !SHARED_REF_MASK;

impl CtfeProvenance {
    /// Returns the `AllocId` of this provenance.
    #[inline(always)]
    pub fn alloc_id(self) -> AllocId {
        AllocId(NonZero::new(self.0.get() & ALLOC_ID_MASK).unwrap())
    }

    /// Returns whether this provenance is immutable.
    #[inline]
    pub fn immutable(self) -> bool {
        self.0.get() & IMMUTABLE_MASK != 0
    }

    /// Returns whether this provenance is derived from a shared reference.
    #[inline]
    pub fn shared_ref(self) -> bool {
        self.0.get() & SHARED_REF_MASK != 0
    }

    pub fn into_parts(self) -> (AllocId, bool, bool) {
        (self.alloc_id(), self.immutable(), self.shared_ref())
    }

    pub fn from_parts((alloc_id, immutable, shared_ref): (AllocId, bool, bool)) -> Self {
        let prov = CtfeProvenance::from(alloc_id);
        if immutable {
            // This sets both flags, so we don't even have to check `shared_ref`.
            prov.as_immutable()
        } else if shared_ref {
            prov.as_shared_ref()
        } else {
            prov
        }
    }

    /// Returns an immutable version of this provenance.
    #[inline]
    pub fn as_immutable(self) -> Self {
        CtfeProvenance(self.0 | IMMUTABLE_MASK | SHARED_REF_MASK)
    }

    /// Returns a "shared reference" (but not necessarily immutable!) version of this provenance.
    #[inline]
    pub fn as_shared_ref(self) -> Self {
        CtfeProvenance(self.0 | SHARED_REF_MASK)
    }
}

impl Provenance for CtfeProvenance {
    // With the `AllocId` as provenance, the `offset` is interpreted *relative to the allocation*,
    // so ptr-to-int casts are not possible (since we do not know the global physical offset).
    const OFFSET_IS_ADDR: bool = false;

    // `CtfeProvenance` does not implement wildcard provenance.
    const WILDCARD: Option<Self> = None;

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print AllocId.
        fmt::Debug::fmt(&ptr.provenance.alloc_id(), f)?; // propagates `alternate` flag
        // Print offset only if it is non-zero.
        if ptr.offset.bytes() > 0 {
            write!(f, "+{:#x}", ptr.offset.bytes())?;
        }
        // Print immutable status.
        if ptr.provenance.immutable() {
            write!(f, "<imm>")?;
        }
        Ok(())
    }

    fn get_alloc_id(self) -> Option<AllocId> {
        Some(self.alloc_id())
    }

    fn join(left: Self, right: Self) -> Option<Self> {
        if left == right { Some(left) } else { None }
    }
}

// We also need this impl so that one can debug-print `Pointer<AllocId>`
impl Provenance for AllocId {
    // With the `AllocId` as provenance, the `offset` is interpreted *relative to the allocation*,
    // so ptr-to-int casts are not possible (since we do not know the global physical offset).
    const OFFSET_IS_ADDR: bool = false;

    // `AllocId` does not implement wildcard provenance.
    const WILDCARD: Option<Self> = None;

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward `alternate` flag to `alloc_id` printing.
        if f.alternate() {
            write!(f, "{:#?}", ptr.provenance)?;
        } else {
            write!(f, "{:?}", ptr.provenance)?;
        }
        // Print offset only if it is non-zero.
        if ptr.offset.bytes() > 0 {
            write!(f, "+{:#x}", ptr.offset.bytes())?;
        }
        Ok(())
    }

    fn get_alloc_id(self) -> Option<AllocId> {
        Some(self)
    }

    fn join(_left: Self, _right: Self) -> Option<Self> {
        unreachable!()
    }
}

/// Represents a pointer in the Miri engine.
///
/// Pointers are "tagged" with provenance information; typically the `AllocId` they belong to.
#[derive(Copy, Clone, Eq, PartialEq, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub struct Pointer<Prov = CtfeProvenance> {
    pub(super) offset: Size, // kept private to avoid accidental misinterpretation (meaning depends on `Prov` type)
    pub provenance: Prov,
}

static_assert_size!(Pointer, 16);
// `Option<Prov>` pointers are also passed around quite a bit
// (but not stored in permanent machine state).
static_assert_size!(Pointer<Option<CtfeProvenance>>, 16);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Prov: Provenance> fmt::Debug for Pointer<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Provenance::fmt(self, f)
    }
}

impl<Prov: Provenance> fmt::Debug for Pointer<Option<Prov>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.provenance {
            Some(prov) => Provenance::fmt(&Pointer::new(prov, self.offset), f),
            None => write!(f, "{:#x}[noalloc]", self.offset.bytes()),
        }
    }
}

impl<Prov: Provenance> fmt::Display for Pointer<Option<Prov>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.provenance.is_none() && self.offset.bytes() == 0 {
            write!(f, "null pointer")
        } else {
            fmt::Debug::fmt(self, f)
        }
    }
}

/// Produces a `Pointer` that points to the beginning of the `Allocation`.
impl From<AllocId> for Pointer {
    #[inline(always)]
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(alloc_id.into(), Size::ZERO)
    }
}
impl From<CtfeProvenance> for Pointer {
    #[inline(always)]
    fn from(prov: CtfeProvenance) -> Self {
        Pointer::new(prov, Size::ZERO)
    }
}

impl<Prov> From<Pointer<Prov>> for Pointer<Option<Prov>> {
    #[inline(always)]
    fn from(ptr: Pointer<Prov>) -> Self {
        let (prov, offset) = ptr.into_raw_parts();
        Pointer::new(Some(prov), offset)
    }
}

impl<Prov> Pointer<Option<Prov>> {
    /// Convert this pointer that *might* have a provenance into a pointer that *definitely* has a
    /// provenance, or an absolute address.
    ///
    /// This is rarely what you want; call `ptr_try_get_alloc_id` instead.
    pub fn into_pointer_or_addr(self) -> Result<Pointer<Prov>, Size> {
        match self.provenance {
            Some(prov) => Ok(Pointer::new(prov, self.offset)),
            None => Err(self.offset),
        }
    }

    /// Returns the absolute address the pointer points to.
    /// Only works if Prov::OFFSET_IS_ADDR is true!
    pub fn addr(self) -> Size
    where
        Prov: Provenance,
    {
        assert!(Prov::OFFSET_IS_ADDR);
        self.offset
    }

    /// Creates a pointer to the given address, with invalid provenance (i.e., cannot be used for
    /// any memory access).
    #[inline(always)]
    pub fn without_provenance(addr: u64) -> Self {
        Pointer { provenance: None, offset: Size::from_bytes(addr) }
    }

    #[inline(always)]
    pub fn null() -> Self {
        Pointer::without_provenance(0)
    }
}

impl<Prov> Pointer<Prov> {
    #[inline(always)]
    pub fn new(provenance: Prov, offset: Size) -> Self {
        Pointer { provenance, offset }
    }

    /// Obtain the constituents of this pointer. Note that the meaning of the offset depends on the
    /// type `Prov`! This is a low-level function that should only be used when absolutely
    /// necessary. Prefer `prov_and_relative_offset` if possible.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (Prov, Size) {
        (self.provenance, self.offset)
    }

    pub fn map_provenance(self, f: impl FnOnce(Prov) -> Prov) -> Self {
        Pointer { provenance: f(self.provenance), ..self }
    }

    #[inline(always)]
    pub fn wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        let res =
            cx.data_layout().truncate_to_target_usize(self.offset.bytes().wrapping_add(i.bytes()));
        Pointer { offset: Size::from_bytes(res), ..self }
    }

    #[inline(always)]
    pub fn wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        // It's wrapping anyway, so we can just cast to `u64`.
        self.wrapping_offset(Size::from_bytes(i as u64), cx)
    }
}

impl Pointer<CtfeProvenance> {
    /// Return the provenance and relative offset stored in this pointer. Safer alternative to
    /// `into_raw_parts` since the type ensures that the offset is indeed relative.
    #[inline(always)]
    pub fn prov_and_relative_offset(self) -> (CtfeProvenance, Size) {
        (self.provenance, self.offset)
    }
}
