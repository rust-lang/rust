use std::assert_matches::debug_assert_matches;
use std::fmt::{self, Display, Write as _};
use std::sync::LazyLock as Lazy;
use std::{ascii, mem};

use rustc_ast::join_path_idents;
use rustc_ast::tokenstream::TokenTree;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_metadata::rendered_const;
use rustc_middle::mir;
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, TyCtxt, TypeVisitableExt};
use rustc_span::symbol::{Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::{debug, warn};
use {rustc_ast as ast, rustc_hir as hir};

use crate::clean::auto_trait::synthesize_auto_trait_impls;
use crate::clean::blanket_impl::synthesize_blanket_impls;
use crate::clean::render_macro_matchers::render_macro_matcher;
use crate::clean::{
    AssocItemConstraint, AssocItemConstraintKind, Crate, ExternalCrate, Generic, GenericArg,
    GenericArgs, ImportSource, Item, ItemKind, Lifetime, Path, PathSegment, Primitive,
    PrimitiveType, Term, Type, clean_doc_module, clean_middle_const, clean_middle_region,
    clean_middle_ty, inline,
};
use crate::core::DocContext;
use crate::display::Joined as _;

#[cfg(test)]
mod tests;

pub(crate) fn krate(cx: &mut DocContext<'_>) -> Crate {
    let module = crate::visit_ast::RustdocVisitor::new(cx).visit();

    // Clean the crate, translating the entire librustc_ast AST to one that is
    // understood by rustdoc.
    let mut module = clean_doc_module(&module, cx);

    match module.kind {
        ItemKind::ModuleItem(ref module) => {
            for it in &module.items {
                // `compiler_builtins` should be masked too, but we can't apply
                // `#[doc(masked)]` to the injected `extern crate` because it's unstable.
                if cx.tcx.is_compiler_builtins(it.item_id.krate()) {
                    cx.cache.masked_crates.insert(it.item_id.krate());
                } else if it.is_extern_crate()
                    && it.attrs.has_doc_flag(sym::masked)
                    && let Some(def_id) = it.item_id.as_def_id()
                    && let Some(local_def_id) = def_id.as_local()
                    && let Some(cnum) = cx.tcx.extern_mod_stmt_cnum(local_def_id)
                {
                    cx.cache.masked_crates.insert(cnum);
                }
            }
        }
        _ => unreachable!(),
    }

    let local_crate = ExternalCrate { crate_num: LOCAL_CRATE };
    let primitives = local_crate.primitives(cx.tcx);
    let keywords = local_crate.keywords(cx.tcx);
    let documented_attributes = local_crate.documented_attributes(cx.tcx);
    {
        let ItemKind::ModuleItem(m) = &mut module.inner.kind else { unreachable!() };
        m.items.extend(primitives.map(|(def_id, prim)| {
            Item::from_def_id_and_parts(
                def_id,
                Some(prim.as_sym()),
                ItemKind::PrimitiveItem(prim),
                cx,
            )
        }));
        m.items.extend(keywords.map(|(def_id, kw)| {
            Item::from_def_id_and_parts(def_id, Some(kw), ItemKind::KeywordItem, cx)
        }));
        m.items.extend(documented_attributes.into_iter().map(|(def_id, kw)| {
            Item::from_def_id_and_parts(def_id, Some(kw), ItemKind::AttributeItem, cx)
        }));
    }

    Crate { module, external_traits: Box::new(mem::take(&mut cx.external_traits)) }
}

pub(crate) fn clean_middle_generic_args<'tcx>(
    cx: &mut DocContext<'tcx>,
    args: ty::Binder<'tcx, &'tcx [ty::GenericArg<'tcx>]>,
    mut has_self: bool,
    owner: DefId,
) -> ThinVec<GenericArg> {
    let (args, bound_vars) = (args.skip_binder(), args.bound_vars());
    if args.is_empty() {
        // Fast path which avoids executing the query `generics_of`.
        return ThinVec::new();
    }

    // If the container is a trait object type, the arguments won't contain the self type but the
    // generics of the corresponding trait will. In such a case, prepend a dummy self type in order
    // to align the arguments and parameters for the iteration below and to enable us to correctly
    // instantiate the generic parameter default later.
    let generics = cx.tcx.generics_of(owner);
    let args = if !has_self && generics.parent.is_none() && generics.has_self {
        has_self = true;
        [cx.tcx.types.trait_object_dummy_self.into()]
            .into_iter()
            .chain(args.iter().copied())
            .collect::<Vec<_>>()
            .into()
    } else {
        std::borrow::Cow::from(args)
    };

    let mut elision_has_failed_once_before = false;
    let clean_arg = |(index, &arg): (usize, &ty::GenericArg<'tcx>)| {
        // Elide the self type.
        if has_self && index == 0 {
            return None;
        }

        let param = generics.param_at(index, cx.tcx);
        let arg = ty::Binder::bind_with_vars(arg, bound_vars);

        // Elide arguments that coincide with their default.
        if !elision_has_failed_once_before && let Some(default) = param.default_value(cx.tcx) {
            let default = default.instantiate(cx.tcx, args.as_ref());
            if can_elide_generic_arg(arg, arg.rebind(default)) {
                return None;
            }
            elision_has_failed_once_before = true;
        }

        match arg.skip_binder().kind() {
            GenericArgKind::Lifetime(lt) => Some(GenericArg::Lifetime(
                clean_middle_region(lt, cx).unwrap_or(Lifetime::elided()),
            )),
            GenericArgKind::Type(ty) => Some(GenericArg::Type(clean_middle_ty(
                arg.rebind(ty),
                cx,
                None,
                Some(crate::clean::ContainerTy::Regular {
                    ty: owner,
                    args: arg.rebind(args.as_ref()),
                    arg: index,
                }),
            ))),
            GenericArgKind::Const(ct) => {
                Some(GenericArg::Const(Box::new(clean_middle_const(arg.rebind(ct), cx))))
            }
        }
    };

    let offset = if has_self { 1 } else { 0 };
    let mut clean_args = ThinVec::with_capacity(args.len().saturating_sub(offset));
    clean_args.extend(args.iter().enumerate().rev().filter_map(clean_arg));
    clean_args.reverse();
    clean_args
}

/// Check if the generic argument `actual` coincides with the `default` and can therefore be elided.
///
/// This uses a very conservative approach for performance and correctness reasons, meaning for
/// several classes of terms it claims that they cannot be elided even if they theoretically could.
/// This is absolutely fine since it mostly concerns edge cases.
fn can_elide_generic_arg<'tcx>(
    actual: ty::Binder<'tcx, ty::GenericArg<'tcx>>,
    default: ty::Binder<'tcx, ty::GenericArg<'tcx>>,
) -> bool {
    debug_assert_matches!(
        (actual.skip_binder().kind(), default.skip_binder().kind()),
        (ty::GenericArgKind::Lifetime(_), ty::GenericArgKind::Lifetime(_))
            | (ty::GenericArgKind::Type(_), ty::GenericArgKind::Type(_))
            | (ty::GenericArgKind::Const(_), ty::GenericArgKind::Const(_))
    );

    // In practice, we shouldn't have any inference variables at this point.
    // However to be safe, we bail out if we do happen to stumble upon them.
    if actual.has_infer() || default.has_infer() {
        return false;
    }

    // Since we don't properly keep track of bound variables in rustdoc (yet), we don't attempt to
    // make any sense out of escaping bound variables. We simply don't have enough context and it
    // would be incorrect to try to do so anyway.
    if actual.has_escaping_bound_vars() || default.has_escaping_bound_vars() {
        return false;
    }

    // Theoretically we could now check if either term contains (non-escaping) late-bound regions or
    // projections, relate the two using an `InferCtxt` and check if the resulting obligations hold.
    // Having projections means that the terms can potentially be further normalized thereby possibly
    // revealing that they are equal after all. Regarding late-bound regions, they could to be
    // liberated allowing us to consider more types to be equal by ignoring the names of binders
    // (e.g., `for<'a> TYPE<'a>` and `for<'b> TYPE<'b>`).
    //
    // However, we are mostly interested in “reeliding” generic args, i.e., eliding generic args that
    // were originally elided by the user and later filled in by the compiler contrary to eliding
    // arbitrary generic arguments if they happen to semantically coincide with the default (of course,
    // we cannot possibly distinguish these two cases). Therefore and for performance reasons, it
    // suffices to only perform a syntactic / structural check by comparing the memory addresses of
    // the interned arguments.
    actual.skip_binder() == default.skip_binder()
}

fn clean_middle_generic_args_with_constraints<'tcx>(
    cx: &mut DocContext<'tcx>,
    did: DefId,
    has_self: bool,
    mut constraints: ThinVec<AssocItemConstraint>,
    args: ty::Binder<'tcx, GenericArgsRef<'tcx>>,
) -> GenericArgs {
    if cx.tcx.is_trait(did)
        && cx.tcx.trait_def(did).paren_sugar
        && let ty::Tuple(tys) = args.skip_binder().type_at(has_self as usize).kind()
    {
        let inputs = tys
            .iter()
            .map(|ty| clean_middle_ty(args.rebind(ty), cx, None, None))
            .collect::<Vec<_>>()
            .into();
        let output = constraints.pop().and_then(|constraint| match constraint.kind {
            AssocItemConstraintKind::Equality { term: Term::Type(ty) } if !ty.is_unit() => {
                Some(Box::new(ty))
            }
            _ => None,
        });
        return GenericArgs::Parenthesized { inputs, output };
    }

    let args = clean_middle_generic_args(cx, args.map_bound(|args| &args[..]), has_self, did);

    GenericArgs::AngleBracketed { args, constraints }
}

pub(super) fn clean_middle_path<'tcx>(
    cx: &mut DocContext<'tcx>,
    did: DefId,
    has_self: bool,
    constraints: ThinVec<AssocItemConstraint>,
    args: ty::Binder<'tcx, GenericArgsRef<'tcx>>,
) -> Path {
    let def_kind = cx.tcx.def_kind(did);
    let name = cx.tcx.opt_item_name(did).unwrap_or(sym::dummy);
    Path {
        res: Res::Def(def_kind, did),
        segments: thin_vec![PathSegment {
            name,
            args: clean_middle_generic_args_with_constraints(cx, did, has_self, constraints, args),
        }],
    }
}

pub(crate) fn qpath_to_string(p: &hir::QPath<'_>) -> String {
    let segments = match *p {
        hir::QPath::Resolved(_, path) => &path.segments,
        hir::QPath::TypeRelative(_, segment) => return segment.ident.to_string(),
        hir::QPath::LangItem(lang_item, ..) => return lang_item.name().to_string(),
    };

    join_path_idents(segments.iter().map(|seg| seg.ident))
}

pub(crate) fn build_deref_target_impls(
    cx: &mut DocContext<'_>,
    items: &[Item],
    ret: &mut Vec<Item>,
) {
    let tcx = cx.tcx;

    for item in items {
        let target = match item.kind {
            ItemKind::AssocTypeItem(ref t, _) => &t.type_,
            _ => continue,
        };

        if let Some(prim) = target.primitive_type() {
            let _prof_timer = tcx.sess.prof.generic_activity("build_primitive_inherent_impls");
            for did in prim.impls(tcx).filter(|did| !did.is_local()) {
                cx.with_param_env(did, |cx| {
                    inline::build_impl(cx, did, None, ret);
                });
            }
        } else if let Type::Path { path } = target {
            let did = path.def_id();
            if !did.is_local() {
                cx.with_param_env(did, |cx| {
                    inline::build_impls(cx, did, None, ret);
                });
            }
        }
    }
}

pub(crate) fn name_from_pat(p: &hir::Pat<'_>) -> Symbol {
    use rustc_hir::*;
    debug!("trying to get a name from pattern: {p:?}");

    Symbol::intern(&match &p.kind {
        PatKind::Err(_)
        | PatKind::Missing // Let's not perpetuate anon params from Rust 2015; use `_` for them.
        | PatKind::Never
        | PatKind::Range(..)
        | PatKind::Struct(..)
        | PatKind::Wild => {
            return kw::Underscore;
        }
        PatKind::Binding(_, _, ident, _) => return ident.name,
        PatKind::Box(p) | PatKind::Ref(p, _) | PatKind::Guard(p, _) => return name_from_pat(p),
        PatKind::TupleStruct(p, ..) | PatKind::Expr(PatExpr { kind: PatExprKind::Path(p), .. }) => {
            qpath_to_string(p)
        }
        PatKind::Or(pats) => {
            fmt::from_fn(|f| pats.iter().map(|p| name_from_pat(p)).joined(" | ", f)).to_string()
        }
        PatKind::Tuple(elts, _) => {
            format!("({})", fmt::from_fn(|f| elts.iter().map(|p| name_from_pat(p)).joined(", ", f)))
        }
        PatKind::Deref(p) => format!("deref!({})", name_from_pat(p)),
        PatKind::Expr(..) => {
            warn!(
                "tried to get argument name from PatKind::Expr, which is silly in function arguments"
            );
            return Symbol::intern("()");
        }
        PatKind::Slice(begin, mid, end) => {
            fn print_pat(pat: &Pat<'_>, wild: bool) -> impl Display {
                fmt::from_fn(move |f| {
                    if wild {
                        f.write_str("..")?;
                    }
                    name_from_pat(pat).fmt(f)
                })
            }

            format!(
                "[{}]",
                fmt::from_fn(|f| {
                    let begin = begin.iter().map(|p| print_pat(p, false));
                    let mid = mid.map(|p| print_pat(p, true));
                    let end = end.iter().map(|p| print_pat(p, false));
                    begin.chain(mid).chain(end).joined(", ", f)
                })
            )
        }
    })
}

pub(crate) fn print_const(cx: &DocContext<'_>, n: ty::Const<'_>) -> String {
    match n.kind() {
        ty::ConstKind::Unevaluated(ty::UnevaluatedConst { def, args: _ }) => {
            if let Some(def) = def.as_local() {
                rendered_const(cx.tcx, cx.tcx.hir_body_owned_by(def), def)
            } else {
                inline::print_inlined_const(cx.tcx, def)
            }
        }
        // array lengths are obviously usize
        ty::ConstKind::Value(cv) if *cv.ty.kind() == ty::Uint(ty::UintTy::Usize) => {
            cv.valtree.unwrap_leaf().to_string()
        }
        _ => n.to_string(),
    }
}

pub(crate) fn print_evaluated_const(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    with_underscores: bool,
    with_type: bool,
) -> Option<String> {
    tcx.const_eval_poly(def_id).ok().and_then(|val| {
        let ty = tcx.type_of(def_id).instantiate_identity();
        match (val, ty.kind()) {
            (_, &ty::Ref(..)) => None,
            (mir::ConstValue::Scalar(_), &ty::Adt(_, _)) => None,
            (mir::ConstValue::Scalar(_), _) => {
                let const_ = mir::Const::from_value(val, ty);
                Some(print_const_with_custom_print_scalar(tcx, const_, with_underscores, with_type))
            }
            _ => None,
        }
    })
}

fn format_integer_with_underscore_sep(num: u128, is_negative: bool) -> String {
    let num = num.to_string();
    let chars = num.as_ascii().unwrap();
    let mut result = if is_negative { "-".to_string() } else { String::new() };
    result.extend(chars.rchunks(3).rev().intersperse(&[ascii::Char::LowLine]).flatten());
    result
}

fn print_const_with_custom_print_scalar<'tcx>(
    tcx: TyCtxt<'tcx>,
    ct: mir::Const<'tcx>,
    with_underscores: bool,
    with_type: bool,
) -> String {
    // Use a slightly different format for integer types which always shows the actual value.
    // For all other types, fallback to the original `pretty_print_const`.
    match (ct, ct.ty().kind()) {
        (mir::Const::Val(mir::ConstValue::Scalar(int), _), ty::Uint(ui)) => {
            let mut output = if with_underscores {
                format_integer_with_underscore_sep(
                    int.assert_scalar_int().to_bits_unchecked(),
                    false,
                )
            } else {
                int.to_string()
            };
            if with_type {
                output += ui.name_str();
            }
            output
        }
        (mir::Const::Val(mir::ConstValue::Scalar(int), _), ty::Int(i)) => {
            let ty = ct.ty();
            let size = tcx
                .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty))
                .unwrap()
                .size;
            let sign_extended_data = int.assert_scalar_int().to_int(size);
            let mut output = if with_underscores {
                format_integer_with_underscore_sep(
                    sign_extended_data.unsigned_abs(),
                    sign_extended_data.is_negative(),
                )
            } else {
                sign_extended_data.to_string()
            };
            if with_type {
                output += i.name_str();
            }
            output
        }
        _ => ct.to_string(),
    }
}

pub(crate) fn is_literal_expr(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    if let hir::Node::Expr(expr) = tcx.hir_node(hir_id) {
        if let hir::ExprKind::Lit(_) = &expr.kind {
            return true;
        }

        if let hir::ExprKind::Unary(hir::UnOp::Neg, expr) = &expr.kind
            && let hir::ExprKind::Lit(_) = &expr.kind
        {
            return true;
        }
    }

    false
}

/// Given a type Path, resolve it to a Type using the TyCtxt
pub(crate) fn resolve_type(cx: &mut DocContext<'_>, path: Path) -> Type {
    debug!("resolve_type({path:?})");

    match path.res {
        Res::PrimTy(p) => Primitive(PrimitiveType::from(p)),
        Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } if path.segments.len() == 1 => {
            Type::SelfTy
        }
        Res::Def(DefKind::TyParam, _) if path.segments.len() == 1 => Generic(path.segments[0].name),
        _ => {
            let _ = register_res(cx, path.res);
            Type::Path { path }
        }
    }
}

pub(crate) fn synthesize_auto_trait_and_blanket_impls(
    cx: &mut DocContext<'_>,
    item_def_id: DefId,
) -> impl Iterator<Item = Item> + use<> {
    let auto_impls = cx
        .sess()
        .prof
        .generic_activity("synthesize_auto_trait_impls")
        .run(|| synthesize_auto_trait_impls(cx, item_def_id));
    let blanket_impls = cx
        .sess()
        .prof
        .generic_activity("synthesize_blanket_impls")
        .run(|| synthesize_blanket_impls(cx, item_def_id));
    auto_impls.into_iter().chain(blanket_impls)
}

/// If `res` has a documentation page associated, store it in the cache.
///
/// This is later used by [`href()`] to determine the HTML link for the item.
///
/// [`href()`]: crate::html::format::href
pub(crate) fn register_res(cx: &mut DocContext<'_>, res: Res) -> DefId {
    use DefKind::*;
    debug!("register_res({res:?})");

    let (kind, did) = match res {
        Res::Def(
            kind @ (AssocTy
            | AssocFn
            | AssocConst
            | Variant
            | Fn
            | TyAlias
            | Enum
            | Trait
            | Struct
            | Union
            | Mod
            | ForeignTy
            | Const
            | Static { .. }
            | Macro(..)
            | TraitAlias),
            did,
        ) => (kind.into(), did),

        _ => panic!("register_res: unexpected {res:?}"),
    };
    if did.is_local() {
        return did;
    }
    inline::record_extern_fqn(cx, did, kind);
    did
}

pub(crate) fn resolve_use_source(cx: &mut DocContext<'_>, path: Path) -> ImportSource {
    ImportSource {
        did: if path.res.opt_def_id().is_none() { None } else { Some(register_res(cx, path.res)) },
        path,
    }
}

pub(crate) fn enter_impl_trait<'tcx, F, R>(cx: &mut DocContext<'tcx>, f: F) -> R
where
    F: FnOnce(&mut DocContext<'tcx>) -> R,
{
    let old_bounds = mem::take(&mut cx.impl_trait_bounds);
    let r = f(cx);
    assert!(cx.impl_trait_bounds.is_empty());
    cx.impl_trait_bounds = old_bounds;
    r
}

/// Find the nearest parent module of a [`DefId`].
pub(crate) fn find_nearest_parent_module(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    if def_id.is_top_level_module() {
        // The crate root has no parent. Use it as the root instead.
        Some(def_id)
    } else {
        let mut current = def_id;
        // The immediate parent might not always be a module.
        // Find the first parent which is.
        while let Some(parent) = tcx.opt_parent(current) {
            if tcx.def_kind(parent) == DefKind::Mod {
                return Some(parent);
            }
            current = parent;
        }
        None
    }
}

/// Checks for the existence of `hidden` in the attribute below if `flag` is `sym::hidden`:
///
/// ```
/// #[doc(hidden)]
/// pub fn foo() {}
/// ```
///
/// This function exists because it runs on `hir::Attributes` whereas the other is a
/// `clean::Attributes` method.
pub(crate) fn has_doc_flag(tcx: TyCtxt<'_>, did: DefId, flag: Symbol) -> bool {
    attrs_have_doc_flag(tcx.get_attrs(did, sym::doc), flag)
}

pub(crate) fn attrs_have_doc_flag<'a>(
    mut attrs: impl Iterator<Item = &'a hir::Attribute>,
    flag: Symbol,
) -> bool {
    attrs.any(|attr| attr.meta_item_list().is_some_and(|l| ast::attr::list_contains_name(&l, flag)))
}

/// A link to `doc.rust-lang.org` that includes the channel name. Use this instead of manual links
/// so that the channel is consistent.
///
/// Set by `bootstrap::Builder::doc_rust_lang_org_channel` in order to keep tests passing on beta/stable.
pub(crate) const DOC_RUST_LANG_ORG_VERSION: &str = env!("DOC_RUST_LANG_ORG_CHANNEL");
pub(crate) static RUSTDOC_VERSION: Lazy<&'static str> =
    Lazy::new(|| DOC_RUST_LANG_ORG_VERSION.rsplit('/').find(|c| !c.is_empty()).unwrap());

/// Render a sequence of macro arms in a format suitable for displaying to the user
/// as part of an item declaration.
fn render_macro_arms<'a>(
    tcx: TyCtxt<'_>,
    matchers: impl Iterator<Item = &'a TokenTree>,
    arm_delim: &str,
) -> String {
    let mut out = String::new();
    for matcher in matchers {
        writeln!(
            out,
            "    {matcher} => {{ ... }}{arm_delim}",
            matcher = render_macro_matcher(tcx, matcher),
        )
        .unwrap();
    }
    out
}

pub(super) fn display_macro_source(
    cx: &mut DocContext<'_>,
    name: Symbol,
    def: &ast::MacroDef,
) -> String {
    // Extract the spans of all matchers. They represent the "interface" of the macro.
    let matchers = def.body.tokens.chunks(4).map(|arm| &arm[0]);

    if def.macro_rules {
        format!("macro_rules! {name} {{\n{arms}}}", arms = render_macro_arms(cx.tcx, matchers, ";"))
    } else {
        if matchers.len() <= 1 {
            format!(
                "macro {name}{matchers} {{\n    ...\n}}",
                matchers = matchers
                    .map(|matcher| render_macro_matcher(cx.tcx, matcher))
                    .collect::<String>(),
            )
        } else {
            format!("macro {name} {{\n{arms}}}", arms = render_macro_arms(cx.tcx, matchers, ","))
        }
    }
}

pub(crate) fn inherits_doc_hidden(
    tcx: TyCtxt<'_>,
    mut def_id: LocalDefId,
    stop_at: Option<LocalDefId>,
) -> bool {
    while let Some(id) = tcx.opt_local_parent(def_id) {
        if let Some(stop_at) = stop_at
            && id == stop_at
        {
            return false;
        }
        def_id = id;
        if tcx.is_doc_hidden(def_id.to_def_id()) {
            return true;
        } else if matches!(
            tcx.hir_node_by_def_id(def_id),
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Impl(_), .. })
        ) {
            // `impl` blocks stand a bit on their own: unless they have `#[doc(hidden)]` directly
            // on them, they don't inherit it from the parent context.
            return false;
        }
    }
    false
}

#[inline]
pub(crate) fn should_ignore_res(res: Res) -> bool {
    matches!(res, Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..))
}
