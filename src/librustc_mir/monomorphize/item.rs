use rustc::hir::def_id::LOCAL_CRATE;
use rustc::mir::mono::MonoItem;
use rustc::session::config::OptLevel;
use rustc::ty::{self, TyCtxt, Instance};
use rustc::ty::subst::InternalSubsts;
use rustc::ty::print::obsolete::DefPathBasedNames;
use syntax::attr::InlineAttr;
use std::fmt;
use rustc::mir::mono::Linkage;
use syntax_pos::symbol::InternedString;
use syntax::source_map::Span;

/// Describes how a monomorphization will be instantiated in object files.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum InstantiationMode {
    /// There will be exactly one instance of the given MonoItem. It will have
    /// external linkage so that it can be linked to from other codegen units.
    GloballyShared {
        /// In some compilation scenarios we may decide to take functions that
        /// are typically `LocalCopy` and instead move them to `GloballyShared`
        /// to avoid codegenning them a bunch of times. In this situation,
        /// however, our local copy may conflict with other crates also
        /// inlining the same function.
        ///
        /// This flag indicates that this situation is occurring, and informs
        /// symbol name calculation that some extra mangling is needed to
        /// avoid conflicts. Note that this may eventually go away entirely if
        /// ThinLTO enables us to *always* have a globally shared instance of a
        /// function within one crate's compilation.
        may_conflict: bool,
    },

    /// Each codegen unit containing a reference to the given MonoItem will
    /// have its own private copy of the function (with internal linkage).
    LocalCopy,
}

pub trait MonoItemExt<'tcx>: fmt::Debug {
    fn as_mono_item(&self) -> &MonoItem<'tcx>;

    fn is_generic_fn(&self) -> bool {
        match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => {
                instance.substs.non_erasable_generics().next().is_some()
            }
            MonoItem::Static(..) |
            MonoItem::GlobalAsm(..) => false,
        }
    }

    fn symbol_name(&self, tcx: TyCtxt<'tcx>) -> ty::SymbolName {
        match *self.as_mono_item() {
            MonoItem::Fn(instance) => tcx.symbol_name(instance),
            MonoItem::Static(def_id) => {
                tcx.symbol_name(Instance::mono(tcx, def_id))
            }
            MonoItem::GlobalAsm(hir_id) => {
                let def_id = tcx.hir().local_def_id_from_hir_id(hir_id);
                ty::SymbolName {
                    name: InternedString::intern(&format!("global_asm_{:?}", def_id))
                }
            }
        }
    }
    fn instantiation_mode(&self, tcx: TyCtxt<'tcx>) -> InstantiationMode {
        let inline_in_all_cgus =
            tcx.sess.opts.debugging_opts.inline_in_all_cgus.unwrap_or_else(|| {
                tcx.sess.opts.optimize != OptLevel::No
            }) && !tcx.sess.opts.cg.link_dead_code;

        match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => {
                let entry_def_id = tcx.entry_fn(LOCAL_CRATE).map(|(id, _)| id);
                // If this function isn't inlined or otherwise has explicit
                // linkage, then we'll be creating a globally shared version.
                if self.explicit_linkage(tcx).is_some() ||
                    !instance.def.requires_local(tcx) ||
                    Some(instance.def_id()) == entry_def_id
                {
                    return InstantiationMode::GloballyShared  { may_conflict: false }
                }

                // At this point we don't have explicit linkage and we're an
                // inlined function. If we're inlining into all CGUs then we'll
                // be creating a local copy per CGU
                if inline_in_all_cgus {
                    return InstantiationMode::LocalCopy
                }

                // Finally, if this is `#[inline(always)]` we're sure to respect
                // that with an inline copy per CGU, but otherwise we'll be
                // creating one copy of this `#[inline]` function which may
                // conflict with upstream crates as it could be an exported
                // symbol.
                match tcx.codegen_fn_attrs(instance.def_id()).inline {
                    InlineAttr::Always => InstantiationMode::LocalCopy,
                    _ => {
                        InstantiationMode::GloballyShared  { may_conflict: true }
                    }
                }
            }
            MonoItem::Static(..) |
            MonoItem::GlobalAsm(..) => {
                InstantiationMode::GloballyShared { may_conflict: false }
            }
        }
    }

    fn explicit_linkage(&self, tcx: TyCtxt<'tcx>) -> Option<Linkage> {
        let def_id = match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => instance.def_id(),
            MonoItem::Static(def_id) => def_id,
            MonoItem::GlobalAsm(..) => return None,
        };

        let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);
        codegen_fn_attrs.linkage
    }

    /// Returns `true` if this instance is instantiable - whether it has no unsatisfied
    /// predicates.
    ///
    /// In order to codegen an item, all of its predicates must hold, because
    /// otherwise the item does not make sense. Type-checking ensures that
    /// the predicates of every item that is *used by* a valid item *do*
    /// hold, so we can rely on that.
    ///
    /// However, we codegen collector roots (reachable items) and functions
    /// in vtables when they are seen, even if they are not used, and so they
    /// might not be instantiable. For example, a programmer can define this
    /// public function:
    ///
    ///     pub fn foo<'a>(s: &'a mut ()) where &'a mut (): Clone {
    ///         <&mut () as Clone>::clone(&s);
    ///     }
    ///
    /// That function can't be codegened, because the method `<&mut () as Clone>::clone`
    /// does not exist. Luckily for us, that function can't ever be used,
    /// because that would require for `&'a mut (): Clone` to hold, so we
    /// can just not emit any code, or even a linker reference for it.
    ///
    /// Similarly, if a vtable method has such a signature, and therefore can't
    /// be used, we can just not emit it and have a placeholder (a null pointer,
    /// which will never be accessed) in its place.
    fn is_instantiable(&self, tcx: TyCtxt<'tcx>) -> bool {
        debug!("is_instantiable({:?})", self);
        let (def_id, substs) = match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => (instance.def_id(), instance.substs),
            MonoItem::Static(def_id) => (def_id, InternalSubsts::empty()),
            // global asm never has predicates
            MonoItem::GlobalAsm(..) => return true
        };

        tcx.substitute_normalize_and_test_predicates((def_id, &substs))
    }

    fn to_string(&self, tcx: TyCtxt<'tcx>, debug: bool) -> String {
        return match *self.as_mono_item() {
            MonoItem::Fn(instance) => {
                to_string_internal(tcx, "fn ", instance, debug)
            },
            MonoItem::Static(def_id) => {
                let instance = Instance::new(def_id, tcx.intern_substs(&[]));
                to_string_internal(tcx, "static ", instance, debug)
            },
            MonoItem::GlobalAsm(..) => {
                "global_asm".to_string()
            }
        };

        fn to_string_internal<'a, 'tcx>(
            tcx: TyCtxt<'tcx>,
            prefix: &str,
            instance: Instance<'tcx>,
            debug: bool,
        ) -> String {
            let mut result = String::with_capacity(32);
            result.push_str(prefix);
            let printer = DefPathBasedNames::new(tcx, false, false);
            printer.push_instance_as_string(instance, &mut result, debug);
            result
        }
    }

    fn local_span(&self, tcx: TyCtxt<'tcx>) -> Option<Span> {
        match *self.as_mono_item() {
            MonoItem::Fn(Instance { def, .. }) => {
                tcx.hir().as_local_hir_id(def.def_id())
            }
            MonoItem::Static(def_id) => {
                tcx.hir().as_local_hir_id(def_id)
            }
            MonoItem::GlobalAsm(hir_id) => {
                Some(hir_id)
            }
        }.map(|hir_id| tcx.hir().span(hir_id))
    }
}

impl MonoItemExt<'tcx> for MonoItem<'tcx> {
    fn as_mono_item(&self) -> &MonoItem<'tcx> {
        self
    }
}
