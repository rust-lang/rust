use crate::clean::auto_trait::AutoTraitFinder;
use crate::clean::blanket_impl::BlanketImplFinder;
use crate::clean::{
    inline, Clean, Crate, ExternalCrate, Generic, GenericArg, GenericArgs, ImportSource, Item,
    ItemKind, Lifetime, Path, PathSegment, Primitive, PrimitiveType, ResolvedPath, Type,
    TypeBinding, Visibility,
};
use crate::core::DocContext;
use crate::formats::item_type::ItemType;

use rustc_ast as ast;
use rustc_ast::tokenstream::TokenTree;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, DefIdTree, TyCtxt};
use rustc_span::symbol::{kw, sym, Symbol};
use std::fmt::Write as _;
use std::mem;

#[cfg(test)]
mod tests;

crate fn krate(cx: &mut DocContext<'_>) -> Crate {
    use crate::visit_lib::LibEmbargoVisitor;

    let module = crate::visit_ast::RustdocVisitor::new(cx).visit();

    let mut externs = Vec::new();
    for &cnum in cx.tcx.crates(()) {
        externs.push(ExternalCrate { crate_num: cnum });
        // Analyze doc-reachability for extern items
        LibEmbargoVisitor::new(cx).visit_lib(cnum);
    }

    // Clean the crate, translating the entire librustc_ast AST to one that is
    // understood by rustdoc.
    let mut module = module.clean(cx);

    match *module.kind {
        ItemKind::ModuleItem(ref module) => {
            for it in &module.items {
                // `compiler_builtins` should be masked too, but we can't apply
                // `#[doc(masked)]` to the injected `extern crate` because it's unstable.
                if it.is_extern_crate()
                    && (it.attrs.has_doc_flag(sym::masked)
                        || cx.tcx.is_compiler_builtins(it.def_id.krate()))
                {
                    cx.cache.masked_crates.insert(it.def_id.krate());
                }
            }
        }
        _ => unreachable!(),
    }

    let local_crate = ExternalCrate { crate_num: LOCAL_CRATE };
    let primitives = local_crate.primitives(cx.tcx);
    let keywords = local_crate.keywords(cx.tcx);
    {
        let m = match *module.kind {
            ItemKind::ModuleItem(ref mut m) => m,
            _ => unreachable!(),
        };
        m.items.extend(primitives.iter().map(|&(def_id, prim)| {
            Item::from_def_id_and_parts(
                def_id,
                Some(prim.as_sym()),
                ItemKind::PrimitiveItem(prim),
                cx,
            )
        }));
        m.items.extend(keywords.into_iter().map(|(def_id, kw)| {
            Item::from_def_id_and_parts(def_id, Some(kw), ItemKind::KeywordItem(kw), cx)
        }));
    }

    Crate {
        module,
        externs,
        primitives,
        external_traits: cx.external_traits.clone(),
        collapsed: false,
    }
}

fn external_generic_args(
    cx: &mut DocContext<'_>,
    did: DefId,
    has_self: bool,
    bindings: Vec<TypeBinding>,
    substs: SubstsRef<'_>,
) -> GenericArgs {
    let mut skip_self = has_self;
    let mut ty_kind = None;
    let args: Vec<_> = substs
        .iter()
        .filter_map(|kind| match kind.unpack() {
            GenericArgKind::Lifetime(lt) => match lt {
                ty::ReLateBound(_, ty::BoundRegion { kind: ty::BrAnon(_), .. }) => {
                    Some(GenericArg::Lifetime(Lifetime::elided()))
                }
                _ => lt.clean(cx).map(GenericArg::Lifetime),
            },
            GenericArgKind::Type(_) if skip_self => {
                skip_self = false;
                None
            }
            GenericArgKind::Type(ty) => {
                ty_kind = Some(ty.kind());
                Some(GenericArg::Type(ty.clean(cx)))
            }
            GenericArgKind::Const(ct) => Some(GenericArg::Const(Box::new(ct.clean(cx)))),
        })
        .collect();

    if cx.tcx.fn_trait_kind_from_lang_item(did).is_some() {
        let inputs = match ty_kind.unwrap() {
            ty::Tuple(tys) => tys.iter().map(|t| t.expect_ty().clean(cx)).collect(),
            _ => return GenericArgs::AngleBracketed { args, bindings },
        };
        let output = None;
        // FIXME(#20299) return type comes from a projection now
        // match types[1].kind {
        //     ty::Tuple(ref v) if v.is_empty() => None, // -> ()
        //     _ => Some(types[1].clean(cx))
        // };
        GenericArgs::Parenthesized { inputs, output }
    } else {
        GenericArgs::AngleBracketed { args, bindings }
    }
}

pub(super) fn external_path(
    cx: &mut DocContext<'_>,
    did: DefId,
    has_self: bool,
    bindings: Vec<TypeBinding>,
    substs: SubstsRef<'_>,
) -> Path {
    let def_kind = cx.tcx.def_kind(did);
    let name = cx.tcx.item_name(did);
    Path {
        res: Res::Def(def_kind, did),
        segments: vec![PathSegment {
            name,
            args: external_generic_args(cx, did, has_self, bindings, substs),
        }],
    }
}

/// Remove the generic arguments from a path.
crate fn strip_path_generics(path: Path) -> Path {
    let segments = path
        .segments
        .iter()
        .map(|s| PathSegment {
            name: s.name,
            args: GenericArgs::AngleBracketed { args: vec![], bindings: vec![] },
        })
        .collect();

    Path { res: path.res, segments }
}

crate fn qpath_to_string(p: &hir::QPath<'_>) -> String {
    let segments = match *p {
        hir::QPath::Resolved(_, path) => &path.segments,
        hir::QPath::TypeRelative(_, segment) => return segment.ident.to_string(),
        hir::QPath::LangItem(lang_item, ..) => return lang_item.name().to_string(),
    };

    let mut s = String::new();
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            s.push_str("::");
        }
        if seg.ident.name != kw::PathRoot {
            s.push_str(&seg.ident.as_str());
        }
    }
    s
}

crate fn build_deref_target_impls(cx: &mut DocContext<'_>, items: &[Item], ret: &mut Vec<Item>) {
    let tcx = cx.tcx;

    for item in items {
        let target = match *item.kind {
            ItemKind::TypedefItem(ref t, true) => &t.type_,
            _ => continue,
        };

        if let Some(prim) = target.primitive_type() {
            for &did in prim.impls(tcx).iter().filter(|did| !did.is_local()) {
                inline::build_impl(cx, None, did, None, ret);
            }
        } else if let ResolvedPath { did, .. } = *target {
            if !did.is_local() {
                inline::build_impls(cx, None, did, None, ret);
            }
        }
    }
}

crate fn name_from_pat(p: &hir::Pat<'_>) -> Symbol {
    use rustc_hir::*;
    debug!("trying to get a name from pattern: {:?}", p);

    Symbol::intern(&match p.kind {
        PatKind::Wild | PatKind::Struct(..) => return kw::Underscore,
        PatKind::Binding(_, _, ident, _) => return ident.name,
        PatKind::TupleStruct(ref p, ..) | PatKind::Path(ref p) => qpath_to_string(p),
        PatKind::Or(pats) => {
            pats.iter().map(|p| name_from_pat(p).to_string()).collect::<Vec<String>>().join(" | ")
        }
        PatKind::Tuple(elts, _) => format!(
            "({})",
            elts.iter().map(|p| name_from_pat(p).to_string()).collect::<Vec<String>>().join(", ")
        ),
        PatKind::Box(p) => return name_from_pat(&*p),
        PatKind::Ref(p, _) => return name_from_pat(&*p),
        PatKind::Lit(..) => {
            warn!(
                "tried to get argument name from PatKind::Lit, which is silly in function arguments"
            );
            return Symbol::intern("()");
        }
        PatKind::Range(..) => return kw::Underscore,
        PatKind::Slice(begin, ref mid, end) => {
            let begin = begin.iter().map(|p| name_from_pat(p).to_string());
            let mid = mid.as_ref().map(|p| format!("..{}", name_from_pat(&**p))).into_iter();
            let end = end.iter().map(|p| name_from_pat(p).to_string());
            format!("[{}]", begin.chain(mid).chain(end).collect::<Vec<_>>().join(", "))
        }
    })
}

crate fn print_const(cx: &DocContext<'_>, n: &'tcx ty::Const<'_>) -> String {
    match n.val {
        ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs_: _, promoted }) => {
            let mut s = if let Some(def) = def.as_local() {
                let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def.did);
                print_const_expr(cx.tcx, cx.tcx.hir().body_owned_by(hir_id))
            } else {
                inline::print_inlined_const(cx.tcx, def.did)
            };
            if let Some(promoted) = promoted {
                s.push_str(&format!("::{:?}", promoted))
            }
            s
        }
        _ => {
            let mut s = n.to_string();
            // array lengths are obviously usize
            if s.ends_with("_usize") {
                let n = s.len() - "_usize".len();
                s.truncate(n);
                if s.ends_with(": ") {
                    let n = s.len() - ": ".len();
                    s.truncate(n);
                }
            }
            s
        }
    }
}

crate fn print_evaluated_const(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    tcx.const_eval_poly(def_id).ok().and_then(|val| {
        let ty = tcx.type_of(def_id);
        match (val, ty.kind()) {
            (_, &ty::Ref(..)) => None,
            (ConstValue::Scalar(_), &ty::Adt(_, _)) => None,
            (ConstValue::Scalar(_), _) => {
                let const_ = ty::Const::from_value(tcx, val, ty);
                Some(print_const_with_custom_print_scalar(tcx, const_))
            }
            _ => None,
        }
    })
}

fn format_integer_with_underscore_sep(num: &str) -> String {
    let num_chars: Vec<_> = num.chars().collect();
    let mut num_start_index = if num_chars.get(0) == Some(&'-') { 1 } else { 0 };
    let chunk_size = match num[num_start_index..].as_bytes() {
        [b'0', b'b' | b'x', ..] => {
            num_start_index += 2;
            4
        }
        [b'0', b'o', ..] => {
            num_start_index += 2;
            let remaining_chars = num_chars.len() - num_start_index;
            if remaining_chars <= 6 {
                // don't add underscores to Unix permissions like 0755 or 100755
                return num.to_string();
            }
            3
        }
        _ => 3,
    };

    num_chars[..num_start_index]
        .iter()
        .chain(num_chars[num_start_index..].rchunks(chunk_size).rev().intersperse(&['_']).flatten())
        .collect()
}

fn print_const_with_custom_print_scalar(tcx: TyCtxt<'_>, ct: &'tcx ty::Const<'tcx>) -> String {
    // Use a slightly different format for integer types which always shows the actual value.
    // For all other types, fallback to the original `pretty_print_const`.
    match (ct.val, ct.ty.kind()) {
        (ty::ConstKind::Value(ConstValue::Scalar(int)), ty::Uint(ui)) => {
            format!("{}{}", format_integer_with_underscore_sep(&int.to_string()), ui.name_str())
        }
        (ty::ConstKind::Value(ConstValue::Scalar(int)), ty::Int(i)) => {
            let ty = tcx.lift(ct.ty).unwrap();
            let size = tcx.layout_of(ty::ParamEnv::empty().and(ty)).unwrap().size;
            let data = int.assert_bits(size);
            let sign_extended_data = size.sign_extend(data) as i128;

            format!(
                "{}{}",
                format_integer_with_underscore_sep(&sign_extended_data.to_string()),
                i.name_str()
            )
        }
        _ => ct.to_string(),
    }
}

crate fn is_literal_expr(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    if let hir::Node::Expr(expr) = tcx.hir().get(hir_id) {
        if let hir::ExprKind::Lit(_) = &expr.kind {
            return true;
        }

        if let hir::ExprKind::Unary(hir::UnOp::Neg, expr) = &expr.kind {
            if let hir::ExprKind::Lit(_) = &expr.kind {
                return true;
            }
        }
    }

    false
}

crate fn print_const_expr(tcx: TyCtxt<'_>, body: hir::BodyId) -> String {
    let hir = tcx.hir();
    let value = &hir.body(body).value;

    let snippet = if !value.span.from_expansion() {
        tcx.sess.source_map().span_to_snippet(value.span).ok()
    } else {
        None
    };

    snippet.unwrap_or_else(|| rustc_hir_pretty::id_to_string(&hir, body.hir_id))
}

/// Given a type Path, resolve it to a Type using the TyCtxt
crate fn resolve_type(cx: &mut DocContext<'_>, path: Path) -> Type {
    debug!("resolve_type({:?})", path);

    match path.res {
        Res::PrimTy(p) => Primitive(PrimitiveType::from(p)),
        Res::SelfTy(..) if path.segments.len() == 1 => Generic(kw::SelfUpper),
        Res::Def(DefKind::TyParam, _) if path.segments.len() == 1 => Generic(path.segments[0].name),
        _ => {
            let did = register_res(cx, path.res);
            ResolvedPath { path, did }
        }
    }
}

crate fn get_auto_trait_and_blanket_impls(
    cx: &mut DocContext<'tcx>,
    item_def_id: DefId,
) -> impl Iterator<Item = Item> {
    let auto_impls = cx
        .sess()
        .prof
        .generic_activity("get_auto_trait_impls")
        .run(|| AutoTraitFinder::new(cx).get_auto_trait_impls(item_def_id));
    let blanket_impls = cx
        .sess()
        .prof
        .generic_activity("get_blanket_impls")
        .run(|| BlanketImplFinder { cx }.get_blanket_impls(item_def_id));
    auto_impls.into_iter().chain(blanket_impls)
}

/// If `res` has a documentation page associated, store it in the cache.
///
/// This is later used by [`href()`] to determine the HTML link for the item.
///
/// [`href()`]: crate::html::format::href
crate fn register_res(cx: &mut DocContext<'_>, res: Res) -> DefId {
    use DefKind::*;
    debug!("register_res({:?})", res);

    let (did, kind) = match res {
        Res::Def(DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst, i) => {
            // associated items are documented, but on the page of their parent
            (cx.tcx.parent(i).unwrap(), ItemType::Trait)
        }
        Res::Def(DefKind::Variant, i) => {
            // variant items are documented, but on the page of their parent
            (cx.tcx.parent(i).expect("cannot get parent def id"), ItemType::Enum)
        }
        // Each of these have their own page.
        Res::Def(
            kind
            @
            (Fn | TyAlias | Enum | Trait | Struct | Union | Mod | ForeignTy | Const | Static
            | Macro(..) | TraitAlias),
            i,
        ) => (i, kind.into()),
        // This is part of a trait definition; document the trait.
        Res::SelfTy(Some(trait_def_id), _) => (trait_def_id, ItemType::Trait),
        // This is an inherent impl; it doesn't have its own page.
        Res::SelfTy(None, Some((impl_def_id, _))) => return impl_def_id,
        Res::SelfTy(None, None)
        | Res::PrimTy(_)
        | Res::ToolMod
        | Res::SelfCtor(_)
        | Res::Local(_)
        | Res::NonMacroAttr(_)
        | Res::Err => return res.def_id(),
        Res::Def(
            TyParam | ConstParam | Ctor(..) | ExternCrate | Use | ForeignMod | AnonConst | OpaqueTy
            | Field | LifetimeParam | GlobalAsm | Impl | Closure | Generator,
            id,
        ) => return id,
    };
    if did.is_local() {
        return did;
    }
    inline::record_extern_fqn(cx, did, kind);
    if let ItemType::Trait = kind {
        inline::record_extern_trait(cx, did);
    }
    did
}

crate fn resolve_use_source(cx: &mut DocContext<'_>, path: Path) -> ImportSource {
    ImportSource {
        did: if path.res.opt_def_id().is_none() { None } else { Some(register_res(cx, path.res)) },
        path,
    }
}

crate fn enter_impl_trait<F, R>(cx: &mut DocContext<'_>, f: F) -> R
where
    F: FnOnce(&mut DocContext<'_>) -> R,
{
    let old_bounds = mem::take(&mut cx.impl_trait_bounds);
    let r = f(cx);
    assert!(cx.impl_trait_bounds.is_empty());
    cx.impl_trait_bounds = old_bounds;
    r
}

/// Find the nearest parent module of a [`DefId`].
crate fn find_nearest_parent_module(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    if def_id.is_top_level_module() {
        // The crate root has no parent. Use it as the root instead.
        Some(def_id)
    } else {
        let mut current = def_id;
        // The immediate parent might not always be a module.
        // Find the first parent which is.
        while let Some(parent) = tcx.parent(current) {
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
crate fn has_doc_flag(attrs: ty::Attributes<'_>, flag: Symbol) -> bool {
    attrs.iter().any(|attr| {
        attr.has_name(sym::doc)
            && attr.meta_item_list().map_or(false, |l| rustc_attr::list_contains_name(&l, flag))
    })
}

/// A link to `doc.rust-lang.org` that includes the channel name. Use this instead of manual links
/// so that the channel is consistent.
///
/// Set by `bootstrap::Builder::doc_rust_lang_org_channel` in order to keep tests passing on beta/stable.
crate const DOC_RUST_LANG_ORG_CHANNEL: &str = env!("DOC_RUST_LANG_ORG_CHANNEL");

/// Render a sequence of macro arms in a format suitable for displaying to the user
/// as part of an item declaration.
pub(super) fn render_macro_arms<'a>(
    matchers: impl Iterator<Item = &'a TokenTree>,
    arm_delim: &str,
) -> String {
    let mut out = String::new();
    for matcher in matchers {
        writeln!(out, "    {} => {{ ... }}{}", render_macro_matcher(matcher), arm_delim).unwrap();
    }
    out
}

/// Render a macro matcher in a format suitable for displaying to the user
/// as part of an item declaration.
pub(super) fn render_macro_matcher(matcher: &TokenTree) -> String {
    rustc_ast_pretty::pprust::tt_to_string(matcher)
}

pub(super) fn display_macro_source(
    cx: &mut DocContext<'_>,
    name: Symbol,
    def: &ast::MacroDef,
    def_id: DefId,
    vis: impl Clean<Visibility>,
) -> String {
    let tts: Vec<_> = def.body.inner_tokens().into_trees().collect();
    // Extract the spans of all matchers. They represent the "interface" of the macro.
    let matchers = tts.chunks(4).map(|arm| &arm[0]);

    if def.macro_rules {
        format!("macro_rules! {} {{\n{}}}", name, render_macro_arms(matchers, ";"))
    } else {
        let vis = vis.clean(cx);

        if matchers.len() <= 1 {
            format!(
                "{}macro {}{} {{\n    ...\n}}",
                vis.to_src_with_space(cx.tcx, def_id),
                name,
                matchers.map(render_macro_matcher).collect::<String>(),
            )
        } else {
            format!(
                "{}macro {} {{\n{}}}",
                vis.to_src_with_space(cx.tcx, def_id),
                name,
                render_macro_arms(matchers, ","),
            )
        }
    }
}
