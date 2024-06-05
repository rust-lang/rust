use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::source_map::Spanned;

use crate::ty::{self, Ty, TyCtxt};

use super::ConstOperand;

#[derive(Debug, HashStable, TyEncodable, TyDecodable, Default)]
pub struct RequiredAndMentionedItems<'tcx> {
    /// Constants that are required to evaluate successfully for this MIR to be well-formed.
    /// We hold in this field all the constants we are not able to evaluate yet.
    ///
    /// This is soundness-critical, we make a guarantee that all consts syntactically mentioned in a
    /// function have successfully evaluated if the function ever gets executed at runtime.
    pub required_consts: Vec<ConstOperand<'tcx>>,

    /// Further items that were mentioned in this function and hence *may* become monomorphized,
    /// depending on optimizations. We use this to avoid optimization-dependent compile errors: the
    /// collector recursively traverses all "mentioned" items and evaluates all their
    /// `required_consts`.
    ///
    /// This is *not* soundness-critical and the contents of this list are *not* a stable guarantee.
    /// All that's relevant is that this set is optimization-level-independent, and that it includes
    /// everything that the collector would consider "used". (For example, we currently compute this
    /// set after drop elaboration, so some drop calls that can never be reached are not considered
    /// "mentioned".) See the documentation of `CollectionMode` in
    /// `compiler/rustc_monomorphize/src/collector.rs` for more context.
    pub mentioned_items: Vec<Spanned<MentionedItem<'tcx>>>,
}

/// Some item that needs to monomorphize successfully for a MIR body to be considered well-formed.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum MentionedItem<'tcx> {
    /// A function that gets called. We don't necessarily know its precise type yet, since it can be
    /// hidden behind a generic.
    Fn(Ty<'tcx>),
    /// A type that has its drop shim called.
    Drop(Ty<'tcx>),
    /// Unsizing casts might require vtables, so we have to record them.
    UnsizeCast { source_ty: Ty<'tcx>, target_ty: Ty<'tcx> },
    /// A closure that is coerced to a function pointer.
    Closure(Ty<'tcx>),
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn required_and_mentioned_items(
        self,
        key: ty::InstanceDef<'tcx>,
    ) -> &'tcx RequiredAndMentionedItems<'tcx> {
        match key {
            ty::InstanceDef::Item(id) => self.required_and_mentioned_items_of_item(id),
            ty::InstanceDef::Intrinsic(_)
            | ty::InstanceDef::VTableShim(_)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::FnPtrShim(_, _)
            | ty::InstanceDef::Virtual(_, _)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceDef::CoroutineKindShim { .. }
            | ty::InstanceDef::ThreadLocalShim(_)
            | ty::InstanceDef::DropGlue(_, _)
            | ty::InstanceDef::CloneShim(_, _)
            | ty::InstanceDef::FnPtrAddrShim(_, _)
            | ty::InstanceDef::AsyncDropGlueCtorShim(_, _) => {
                self.required_and_mentioned_items_of_shim(key)
            }
        }
    }
}
