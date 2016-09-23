use memory::{Memory, Pointer};
use error::EvalResult;
use primval::PrimVal;

/// A `Value` represents a single self-contained Rust value.
///
/// A `Value` can either refer to a block of memory inside an allocation (`ByRef`) or to a primitve
/// value held directly, outside of any allocation (`ByVal`).
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ByValPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's trans.
#[derive(Clone, Copy, Debug)]
pub(super) enum Value {
    ByRef(Pointer),
    ByVal(PrimVal),
}

impl Value {
    pub(super) fn read_ptr<'a, 'tcx: 'a>(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, Pointer> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => mem.read_ptr(ptr),
            ByVal(PrimVal::Ptr(ptr)) |
            ByVal(PrimVal::FnPtr(ptr)) => Ok(ptr),
            ByVal(_other) => unimplemented!(),
        }
    }

    pub(super) fn expect_vtable<'a, 'tcx: 'a>(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, Pointer> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => mem.read_ptr(ptr.offset(mem.pointer_size() as isize)),
            ByVal(PrimVal::VtablePtr(_, vtable)) => Ok(vtable),
            _ => unimplemented!(),
        }
    }

    pub(super) fn expect_slice_len<'a, 'tcx: 'a>(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, u64> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => mem.read_usize(ptr.offset(mem.pointer_size() as isize)),
            ByVal(PrimVal::SlicePtr(_, len)) => Ok(len),
            _ => unimplemented!(),
        }
    }

    pub(super) fn expect_fat_ptr_extra<'a, 'tcx: 'a>(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, PrimVal> {
        use self::Value::*;
        match (*self, mem.pointer_size()) {
            (ByRef(ptr), size) => mem.read_ptr(ptr.offset(size as isize)).map(PrimVal::Ptr),
            (ByVal(PrimVal::SlicePtr(_, len)), 8) => Ok(PrimVal::U64(len)),
            (ByVal(PrimVal::SlicePtr(_, len)), 4) => Ok(PrimVal::U32(len as u32)),
            (ByVal(PrimVal::SlicePtr(_, len)), 2) => Ok(PrimVal::U16(len as u16)),
            (ByVal(PrimVal::SlicePtr(_, len)), 1) => Ok(PrimVal::U8(len as u8)),
            (ByVal(PrimVal::VtablePtr(_, ptr)), _) => Ok(PrimVal::Ptr(ptr)),
            _ => unimplemented!(),
        }
    }
}
