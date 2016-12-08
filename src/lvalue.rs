use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::ty::Ty;
use rustc::ty::subst::Substs;

use memory::Pointer;
use eval_context::Value;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Lvalue<'tcx> {
    /// An lvalue referring to a value allocated in the `Memory` system.
    Ptr {
        ptr: Pointer,
        extra: LvalueExtra,
    },

    /// An lvalue referring to a value on the stack. Represented by a stack frame index paired with
    /// a Mir local index.
    Local {
        frame: usize,
        local: mir::Local,
    },

    /// An lvalue referring to a global
    Global(GlobalId<'tcx>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LvalueExtra {
    None,
    Length(u64),
    Vtable(Pointer),
    DowncastVariant(usize),
}

/// Uniquely identifies a specific constant or static.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GlobalId<'tcx> {
    /// For a constant or static, the `DefId` of the item itself.
    /// For a promoted global, the `DefId` of the function they belong to.
    pub(super) def_id: DefId,

    /// For statics and constants this is `Substs::empty()`, so only promoteds and associated
    /// constants actually have something useful here. We could special case statics and constants,
    /// but that would only require more branching when working with constants, and not bring any
    /// real benefits.
    pub(super) substs: &'tcx Substs<'tcx>,

    /// The index for promoted globals within their function's `Mir`.
    pub(super) promoted: Option<mir::Promoted>,
}

#[derive(Copy, Clone, Debug)]
pub struct Global<'tcx> {
    pub(super) data: Option<Value>,
    pub(super) mutable: bool,
    pub(super) ty: Ty<'tcx>,
}

impl<'tcx> Global<'tcx> {
    pub(super) fn uninitialized(ty: Ty<'tcx>) -> Self {
        Global {
            data: None,
            mutable: true,
            ty: ty,
        }
    }
}

