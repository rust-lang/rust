use rustc_abi::FIRST_VARIANT;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir as hir;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::DefKind;
use rustc_hir::find_attr;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, AdtDef, Instance, Ty, TyCtxt};
use rustc_session::declare_lint;
use rustc_span::{Span, Symbol};
use tracing::{debug, instrument};

use crate::lints::{BuiltinClashingExtern, BuiltinClashingExternSub};
use crate::{LintVec, types};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { clashing_extern_declarations, ..*providers };
}

pub(crate) fn get_lints() -> LintVec {
    vec![CLASHING_EXTERN_DECLARATIONS]
}

fn clashing_extern_declarations(tcx: TyCtxt<'_>, (): ()) {
    let mut lint = ClashingExternDeclarations::new();
    for id in tcx.hir_crate_items(()).foreign_items() {
        lint.check_foreign_item(tcx, id);
    }
}

declare_lint! {
    /// The `clashing_extern_declarations` lint detects when an `extern fn`
    /// has been declared with the same name but different types.
    ///
    /// ### Example
    ///
    /// ```rust
    /// mod m {
    ///     unsafe extern "C" {
    ///         fn foo();
    ///     }
    /// }
    ///
    /// unsafe extern "C" {
    ///     fn foo(_: u32);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Because two symbols of the same name cannot be resolved to two
    /// different functions at link time, and one function cannot possibly
    /// have two types, a clashing extern declaration is almost certainly a
    /// mistake. Check to make sure that the `extern` definitions are correct
    /// and equivalent, and possibly consider unifying them in one location.
    ///
    /// This lint does not run between crates because a project may have
    /// dependencies which both rely on the same extern function, but declare
    /// it in a different (but valid) way. For example, they may both declare
    /// an opaque type for one or more of the arguments (which would end up
    /// distinct types), or use types that are valid conversions in the
    /// language the `extern fn` is defined in. In these cases, the compiler
    /// can't say that the clashing declaration is incorrect.
    pub CLASHING_EXTERN_DECLARATIONS,
    Warn,
    "detects when an extern fn has been declared with the same name but different types"
}

struct ClashingExternDeclarations {
    /// Map of function symbol name to the first-seen hir id for that symbol name.. If seen_decls
    /// contains an entry for key K, it means a symbol with name K has been seen by this lint and
    /// the symbol should be reported as a clashing declaration.
    // FIXME: Technically, we could just store a &'tcx str here without issue; however, the
    // `impl_lint_pass` macro doesn't currently support lints parametric over a lifetime.
    seen_decls: UnordMap<Symbol, hir::OwnerId>,
}

/// Differentiate between whether the name for an extern decl came from the link_name attribute or
/// just from declaration itself. This is important because we don't want to report clashes on
/// symbol name if they don't actually clash because one or the other links against a symbol with a
/// different name.
enum SymbolName {
    /// The name of the symbol + the span of the annotation which introduced the link name.
    Link(Symbol, Span),
    /// No link name, so just the name of the symbol.
    Normal(Symbol),
}

impl SymbolName {
    fn get_name(&self) -> Symbol {
        match self {
            SymbolName::Link(s, _) | SymbolName::Normal(s) => *s,
        }
    }
}

impl ClashingExternDeclarations {
    pub(crate) fn new() -> Self {
        ClashingExternDeclarations { seen_decls: Default::default() }
    }

    /// Insert a new foreign item into the seen set. If a symbol with the same name already exists
    /// for the item, return its HirId without updating the set.
    fn insert(&mut self, tcx: TyCtxt<'_>, fi: hir::ForeignItemId) -> Option<hir::OwnerId> {
        let did = fi.owner_id.to_def_id();
        let instance = Instance::new_raw(did, ty::List::identity_for_item(tcx, did));
        let name = Symbol::intern(tcx.symbol_name(instance).name);
        if let Some(&existing_id) = self.seen_decls.get(&name) {
            // Avoid updating the map with the new entry when we do find a collision. We want to
            // make sure we're always pointing to the first definition as the previous declaration.
            // This lets us avoid emitting "knock-on" diagnostics.
            Some(existing_id)
        } else {
            self.seen_decls.insert(name, fi.owner_id)
        }
    }

    #[instrument(level = "trace", skip(self, tcx))]
    fn check_foreign_item<'tcx>(&mut self, tcx: TyCtxt<'tcx>, this_fi: hir::ForeignItemId) {
        let DefKind::Fn = tcx.def_kind(this_fi.owner_id) else { return };
        let Some(existing_did) = self.insert(tcx, this_fi) else { return };

        let existing_decl_ty = tcx.type_of(existing_did).skip_binder();
        let this_decl_ty = tcx.type_of(this_fi.owner_id).instantiate_identity();
        debug!(
            "ClashingExternDeclarations: Comparing existing {:?}: {:?} to this {:?}: {:?}",
            existing_did, existing_decl_ty, this_fi.owner_id, this_decl_ty
        );

        // Check that the declarations match.
        if !structurally_same_type(
            tcx,
            ty::TypingEnv::non_body_analysis(tcx, this_fi.owner_id),
            existing_decl_ty,
            this_decl_ty,
        ) {
            let orig = name_of_extern_decl(tcx, existing_did);

            // Finally, emit the diagnostic.
            let this = tcx.item_name(this_fi.owner_id.to_def_id());
            let orig = orig.get_name();
            let previous_decl_label = get_relevant_span(tcx, existing_did);
            let mismatch_label = get_relevant_span(tcx, this_fi.owner_id);
            let sub =
                BuiltinClashingExternSub { tcx, expected: existing_decl_ty, found: this_decl_ty };
            let decorator = if orig == this {
                BuiltinClashingExtern::SameName {
                    this,
                    orig,
                    previous_decl_label,
                    mismatch_label,
                    sub,
                }
            } else {
                BuiltinClashingExtern::DiffName {
                    this,
                    orig,
                    previous_decl_label,
                    mismatch_label,
                    sub,
                }
            };
            tcx.emit_node_span_lint(
                CLASHING_EXTERN_DECLARATIONS,
                this_fi.hir_id(),
                mismatch_label,
                decorator,
            );
        }
    }
}

/// Get the name of the symbol that's linked against for a given extern declaration. That is,
/// the name specified in a #[link_name = ...] attribute if one was specified, else, just the
/// symbol's name.
fn name_of_extern_decl(tcx: TyCtxt<'_>, fi: hir::OwnerId) -> SymbolName {
    if let Some((overridden_link_name, overridden_link_name_span)) =
        tcx.codegen_fn_attrs(fi).symbol_name.map(|overridden_link_name| {
            // FIXME: Instead of searching through the attributes again to get span
            // information, we could have codegen_fn_attrs also give span information back for
            // where the attribute was defined. However, until this is found to be a
            // bottleneck, this does just fine.
            (
                overridden_link_name,
                find_attr!(tcx.get_all_attrs(fi), AttributeKind::LinkName {span, ..} => *span)
                    .unwrap(),
            )
        })
    {
        SymbolName::Link(overridden_link_name, overridden_link_name_span)
    } else {
        SymbolName::Normal(tcx.item_name(fi.to_def_id()))
    }
}

/// We want to ensure that we use spans for both decls that include where the
/// name was defined, whether that was from the link_name attribute or not.
fn get_relevant_span(tcx: TyCtxt<'_>, fi: hir::OwnerId) -> Span {
    match name_of_extern_decl(tcx, fi) {
        SymbolName::Normal(_) => tcx.def_span(fi),
        SymbolName::Link(_, annot_span) => annot_span,
    }
}

/// Checks whether two types are structurally the same enough that the declarations shouldn't
/// clash. We need this so we don't emit a lint when two modules both declare an extern struct,
/// with the same members (as the declarations shouldn't clash).
fn structurally_same_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
) -> bool {
    let mut seen_types = UnordSet::default();
    let result = structurally_same_type_impl(&mut seen_types, tcx, typing_env, a, b);
    if cfg!(debug_assertions) && result {
        // Sanity-check: must have same ABI, size and alignment.
        // `extern` blocks cannot be generic, so we'll always get a layout here.
        let a_layout = tcx.layout_of(typing_env.as_query_input(a)).unwrap();
        let b_layout = tcx.layout_of(typing_env.as_query_input(b)).unwrap();
        assert_eq!(a_layout.backend_repr, b_layout.backend_repr);
        assert_eq!(a_layout.size, b_layout.size);
        assert_eq!(a_layout.align, b_layout.align);
    }
    result
}

fn structurally_same_type_impl<'tcx>(
    seen_types: &mut UnordSet<(Ty<'tcx>, Ty<'tcx>)>,
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
) -> bool {
    debug!("structurally_same_type_impl(tcx, a = {:?}, b = {:?})", a, b);

    // Given a transparent newtype, reach through and grab the inner
    // type unless the newtype makes the type non-null.
    let non_transparent_ty = |mut ty: Ty<'tcx>| -> Ty<'tcx> {
        loop {
            if let ty::Adt(def, args) = *ty.kind() {
                let is_transparent = def.repr().transparent();
                let is_non_null = types::nonnull_optimization_guaranteed(tcx, def);
                debug!(?ty, is_transparent, is_non_null);
                if is_transparent && !is_non_null {
                    debug_assert_eq!(def.variants().len(), 1);
                    let v = &def.variant(FIRST_VARIANT);
                    // continue with `ty`'s non-ZST field,
                    // otherwise `ty` is a ZST and we can return
                    if let Some(field) = types::transparent_newtype_field(tcx, v) {
                        ty = field.ty(tcx, args);
                        continue;
                    }
                }
            }
            debug!("non_transparent_ty -> {:?}", ty);
            return ty;
        }
    };

    let a = non_transparent_ty(a);
    let b = non_transparent_ty(b);

    if !seen_types.insert((a, b)) {
        // We've encountered a cycle. There's no point going any further -- the types are
        // structurally the same.
        true
    } else if a == b {
        // All nominally-same types are structurally same, too.
        true
    } else {
        // Do a full, depth-first comparison between the two.
        let is_primitive_or_pointer =
            |ty: Ty<'tcx>| ty.is_primitive() || matches!(ty.kind(), ty::RawPtr(..) | ty::Ref(..));

        ensure_sufficient_stack(|| {
            match (a.kind(), b.kind()) {
                (&ty::Adt(a_def, a_gen_args), &ty::Adt(b_def, b_gen_args)) => {
                    // Only `repr(C)` types can be compared structurally.
                    if !(a_def.repr().c() && b_def.repr().c()) {
                        return false;
                    }
                    // If the types differ in their packed-ness, align, or simd-ness they conflict.
                    let repr_characteristica =
                        |def: AdtDef<'tcx>| (def.repr().pack, def.repr().align, def.repr().simd());
                    if repr_characteristica(a_def) != repr_characteristica(b_def) {
                        return false;
                    }

                    // Grab a flattened representation of all fields.
                    let a_fields = a_def.variants().iter().flat_map(|v| v.fields.iter());
                    let b_fields = b_def.variants().iter().flat_map(|v| v.fields.iter());

                    // Perform a structural comparison for each field.
                    a_fields.eq_by(
                        b_fields,
                        |&ty::FieldDef { did: a_did, .. }, &ty::FieldDef { did: b_did, .. }| {
                            structurally_same_type_impl(
                                seen_types,
                                tcx,
                                typing_env,
                                tcx.type_of(a_did).instantiate(tcx, a_gen_args),
                                tcx.type_of(b_did).instantiate(tcx, b_gen_args),
                            )
                        },
                    )
                }
                (ty::Array(a_ty, a_len), ty::Array(b_ty, b_len)) => {
                    // For arrays, we also check the length.
                    a_len == b_len
                        && structurally_same_type_impl(seen_types, tcx, typing_env, *a_ty, *b_ty)
                }
                (ty::Slice(a_ty), ty::Slice(b_ty)) => {
                    structurally_same_type_impl(seen_types, tcx, typing_env, *a_ty, *b_ty)
                }
                (ty::RawPtr(a_ty, a_mutbl), ty::RawPtr(b_ty, b_mutbl)) => {
                    a_mutbl == b_mutbl
                        && structurally_same_type_impl(seen_types, tcx, typing_env, *a_ty, *b_ty)
                }
                (ty::Ref(_a_region, a_ty, a_mut), ty::Ref(_b_region, b_ty, b_mut)) => {
                    // For structural sameness, we don't need the region to be same.
                    a_mut == b_mut
                        && structurally_same_type_impl(seen_types, tcx, typing_env, *a_ty, *b_ty)
                }
                (ty::FnDef(..), ty::FnDef(..)) => {
                    let a_poly_sig = a.fn_sig(tcx);
                    let b_poly_sig = b.fn_sig(tcx);

                    // We don't compare regions, but leaving bound regions around ICEs, so
                    // we erase them.
                    let a_sig = tcx.instantiate_bound_regions_with_erased(a_poly_sig);
                    let b_sig = tcx.instantiate_bound_regions_with_erased(b_poly_sig);

                    (a_sig.abi, a_sig.safety, a_sig.c_variadic)
                        == (b_sig.abi, b_sig.safety, b_sig.c_variadic)
                        && a_sig.inputs().iter().eq_by(b_sig.inputs().iter(), |a, b| {
                            structurally_same_type_impl(seen_types, tcx, typing_env, *a, *b)
                        })
                        && structurally_same_type_impl(
                            seen_types,
                            tcx,
                            typing_env,
                            a_sig.output(),
                            b_sig.output(),
                        )
                }
                (ty::Tuple(..), ty::Tuple(..)) => {
                    // Tuples are not `repr(C)` so these cannot be compared structurally.
                    false
                }
                // For these, it's not quite as easy to define structural-sameness quite so easily.
                // For the purposes of this lint, take the conservative approach and mark them as
                // not structurally same.
                (ty::Dynamic(..), ty::Dynamic(..))
                | (ty::Error(..), ty::Error(..))
                | (ty::Closure(..), ty::Closure(..))
                | (ty::Coroutine(..), ty::Coroutine(..))
                | (ty::CoroutineWitness(..), ty::CoroutineWitness(..))
                | (ty::Alias(ty::Projection, ..), ty::Alias(ty::Projection, ..))
                | (ty::Alias(ty::Inherent, ..), ty::Alias(ty::Inherent, ..))
                | (ty::Alias(ty::Opaque, ..), ty::Alias(ty::Opaque, ..)) => false,

                // These definitely should have been caught above.
                (ty::Bool, ty::Bool)
                | (ty::Char, ty::Char)
                | (ty::Never, ty::Never)
                | (ty::Str, ty::Str) => unreachable!(),

                // An Adt and a primitive or pointer type. This can be FFI-safe if non-null
                // enum layout optimisation is being applied.
                (ty::Adt(..) | ty::Pat(..), _) if is_primitive_or_pointer(b) => {
                    if let Some(a_inner) = types::repr_nullable_ptr(tcx, typing_env, a) {
                        a_inner == b
                    } else {
                        false
                    }
                }
                (_, ty::Adt(..) | ty::Pat(..)) if is_primitive_or_pointer(a) => {
                    if let Some(b_inner) = types::repr_nullable_ptr(tcx, typing_env, b) {
                        b_inner == a
                    } else {
                        false
                    }
                }

                _ => false,
            }
        })
    }
}
