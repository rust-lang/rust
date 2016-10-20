use error::EvalResult;
use memory::{Memory, Pointer};
use primval::{PrimVal, PrimValKind};

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

            ByVal(PrimVal { kind: PrimValKind::Ptr(alloc), bits: offset }) |
            ByVal(PrimVal { kind: PrimValKind::FnPtr(alloc), bits: offset }) => {
                let ptr = Pointer::new(alloc, offset as usize);
                Ok(ptr)
            }

            ByValPair(..) => unimplemented!(),
            ByVal(_other) => unimplemented!(),
        }
    }

    pub(super) fn expect_ptr_vtable_pair(
        &self,
        mem: &Memory<'a, 'tcx>
    ) -> EvalResult<'tcx, (Pointer, Pointer)> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => {
                let ptr = mem.read_ptr(ptr)?;
                let vtable = mem.read_ptr(ptr.offset(mem.pointer_size() as isize))?;
                Ok((ptr, vtable))
            }

            ByValPair(
                PrimVal { kind: PrimValKind::Ptr(ptr_alloc), bits: ptr_offset },
                PrimVal { kind: PrimValKind::Ptr(vtable_alloc), bits: vtable_offset },
            ) => {
                let ptr = Pointer::new(ptr_alloc, ptr_offset as usize);
                let vtable = Pointer::new(vtable_alloc, vtable_offset as usize);
                Ok((ptr, vtable))
            }

            _ => bug!("expected ptr and vtable, got {:?}", self),
        }
    }

    pub(super) fn expect_slice_len(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, u64> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => mem.read_usize(ptr.offset(mem.pointer_size() as isize)),
            ByValPair(_, val) if val.kind.is_int() => Ok(val.bits),
            _ => unimplemented!(),
        }
    }
}
