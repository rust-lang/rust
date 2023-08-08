//! Locals are in a private module as updating `LocalRef::Operand` has to
//! be careful wrt to subtyping. To deal with this we only allow updates by using
//! `FunctionCx::overwrite_local` which handles it automatically.
use crate::mir::{FunctionCx, LocalRef};
use crate::traits::BuilderMethods;
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_middle::ty::print::with_no_trimmed_paths;
use std::ops::{Index, IndexMut};

pub(super) struct Locals<'tcx, V> {
    values: IndexVec<mir::Local, LocalRef<'tcx, V>>,
}

impl<'tcx, V> Index<mir::Local> for Locals<'tcx, V> {
    type Output = LocalRef<'tcx, V>;
    #[inline]
    fn index(&self, index: mir::Local) -> &LocalRef<'tcx, V> {
        &self.values[index]
    }
}

/// To mutate locals, use `FunctionCx::overwrite_local` instead.
impl<'tcx, V, Idx: ?Sized> !IndexMut<Idx> for Locals<'tcx, V> {}

impl<'tcx, V> Locals<'tcx, V> {
    pub(super) fn empty() -> Locals<'tcx, V> {
        Locals { values: IndexVec::default() }
    }

    pub(super) fn indices(&self) -> impl DoubleEndedIterator<Item = mir::Local> + Clone + 'tcx {
        self.values.indices()
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub(super) fn initialize_locals(&mut self, values: Vec<LocalRef<'tcx, Bx::Value>>) {
        assert!(self.locals.values.is_empty());

        for (local, value) in values.into_iter().enumerate() {
            match value {
                LocalRef::Place(_) | LocalRef::UnsizedPlace(_) | LocalRef::PendingOperand => (),
                LocalRef::Operand(op) => {
                    let local = mir::Local::from_usize(local);
                    let expected_ty = self.monomorphize(self.mir.local_decls[local].ty);
                    assert_eq!(expected_ty, op.layout.ty, "unexpected initial operand type");
                }
            }

            self.locals.values.push(value);
        }
    }

    pub(super) fn overwrite_local(
        &mut self,
        local: mir::Local,
        mut value: LocalRef<'tcx, Bx::Value>,
    ) {
        match value {
            LocalRef::Place(_) | LocalRef::UnsizedPlace(_) | LocalRef::PendingOperand => (),
            LocalRef::Operand(ref mut op) => {
                let local_ty = self.monomorphize(self.mir.local_decls[local].ty);
                if local_ty != op.layout.ty {
                    // FIXME(#112651): This can be changed to an ICE afterwards.
                    debug!("updating type of operand due to subtyping");
                    with_no_trimmed_paths!(debug!(?op.layout.ty));
                    with_no_trimmed_paths!(debug!(?local_ty));
                    op.layout.ty = local_ty;
                }
            }
        };

        self.locals.values[local] = value;
    }
}
