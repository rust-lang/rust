use error::EvalResult;
use memory::{Memory, Pointer};
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
pub enum Value {
    ByRef(Pointer),
    ByVal(PrimVal),
    ByValPair(PrimVal, PrimVal),
}

impl<'a, 'tcx: 'a> Value {
    pub(super) fn read_ptr(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, Pointer> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => mem.read_ptr(ptr),
            ByVal(ptr) | ByValPair(ptr, _) => Ok(ptr.to_ptr()),
        }
    }

    pub(super) fn expect_ptr_vtable_pair(
        &self,
        mem: &Memory<'a, 'tcx>
    ) -> EvalResult<'tcx, (Pointer, Pointer)> {
        use self::Value::*;
        match *self {
            ByRef(ref_ptr) => {
                let ptr = mem.read_ptr(ref_ptr)?;
                let vtable = mem.read_ptr(ref_ptr.offset(mem.pointer_size() as isize))?;
                Ok((ptr, vtable))
            }

            ByValPair(ptr, vtable) => Ok((ptr.to_ptr(), vtable.to_ptr())),

            _ => bug!("expected ptr and vtable, got {:?}", self),
        }
    }

    pub(super) fn expect_slice(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, (Pointer, u64)> {
        use self::Value::*;
        match *self {
            ByRef(ref_ptr) => {
                let ptr = mem.read_ptr(ref_ptr)?;
                let len = mem.read_usize(ref_ptr.offset(mem.pointer_size() as isize))?;
                Ok((ptr, len))
            },
            ByValPair(ptr, val) => {
                Ok((ptr.to_ptr(), val.try_as_uint()?))
            },
            _ => unimplemented!(),
        }
    }
}
