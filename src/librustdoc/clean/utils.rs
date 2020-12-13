use crate::clean::auto_trait::AutoTraitFinder;
use crate::clean::blanket_impl::BlanketImplFinder;
use crate::clean::{
    inline, Clean, Crate, ExternalCrate, FnDecl, FnRetTy, Generic, GenericArg, GenericArgs,
    GenericBound, Generics, GetDefId, ImportSource, Item, ItemKind, Lifetime, MacroKind, Path,
    PathSegment, Primitive, PrimitiveType, ResolvedPath, Type, TypeBinding, TypeKind,
    WherePredicate,
};
use crate::core::DocContext;

use itertools::Itertools;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, DefIdTree, Ty};
use rustc_span::symbol::{kw, sym, Symbol};
use std::mem;

crate fn krate(mut cx: &mut DocContext<'_>) -> Crate {
    use crate::visit_lib::LibEmbargoVisitor;

    let krate = cx.tcx.hir().krate();
    let module = crate::visit_ast::RustdocVisitor::new(&mut cx).visit(krate);

    let mut r = cx.renderinfo.get_mut();
    r.deref_trait_did = cx.tcx.lang_items().deref_trait();
    r.deref_mut_trait_did = cx.tcx.lang_items().deref_mut_trait();
    r.owned_box_did = cx.tcx.lang_items().owned_box();

    let mut externs = Vec::new();
    for &cnum in cx.tcx.crates().iter() {
        externs.push((cnum, cnum.clean(cx)));
        // Analyze doc-reachability for extern items
        LibEmbargoVisitor::new(&mut cx).visit_lib(cnum);
    }
    externs.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

    // Clean the crate, translating the entire librustc_ast AST to one that is
    // understood by rustdoc.
    let mut module = module.clean(cx);
    let mut masked_crates = FxHashSet::default();

    match *module.kind {
        ItemKind::ModuleItem(ref module) => {
            for it in &module.items {
                // `compiler_builtins` should be masked too, but we can't apply
                // `#[doc(masked)]` to the injected `extern crate` because it's unstable.
                if it.is_extern_crate()
                    && (it.attrs.has_doc_flag(sym::masked)
                        || cx.tcx.is_compiler_builtins(it.def_id.krate))
                {
                    masked_crates.insert(it.def_id.krate);
                }
            }
        }
        _ => unreachable!(),
    }

    let ExternalCrate { name, src, primitives, keywords, .. } = LOCAL_CRATE.clean(cx);
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
            Item::from_def_id_and_parts(def_id, Some(kw.clone()), ItemKind::KeywordItem(kw), cx)
        }));
    }

    Crate {
        name,
        version: None,
        src,
        module: Some(module),
        externs,
        primitives,
        external_traits: cx.external_traits.clone(),
        masked_crates,
        collapsed: false,
    }
}

fn external_generic_args(
    cx: &DocContext<'_>,
    trait_did: Option<DefId>,
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
                ty::ReLateBound(_, ty::BoundRegion { kind: ty::BrAnon(_) }) => {
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
            GenericArgKind::Const(ct) => Some(GenericArg::Const(ct.clean(cx))),
        })
        .collect();

    match trait_did {
        // Attempt to sugar an external path like Fn<(A, B,), C> to Fn(A, B) -> C
        Some(did) if cx.tcx.fn_trait_kind_from_lang_item(did).is_some() => {
            assert!(ty_kind.is_some());
            let inputs = match ty_kind {
                Some(ty::Tuple(ref tys)) => tys.iter().map(|t| t.expect_ty().clean(cx)).collect(),
                _ => return GenericArgs::AngleBracketed { args, bindings },
            };
            let output = None;
            // FIXME(#20299) return type comes from a projection now
            // match types[1].kind {
            //     ty::Tuple(ref v) if v.is_empty() => None, // -> ()
            //     _ => Some(types[1].clean(cx))
            // };
            GenericArgs::Parenthesized { inputs, output }
        }
        _ => GenericArgs::AngleBracketed { args, bindings },
    }
}

// trait_did should be set to a trait's DefId if called on a TraitRef, in order to sugar
// from Fn<(A, B,), C> to Fn(A, B) -> C
pub(super) fn external_path(
    cx: &DocContext<'_>,
    name: Symbol,
    trait_did: Option<DefId>,
    has_self: bool,
    bindings: Vec<TypeBinding>,
    substs: SubstsRef<'_>,
) -> Path {
    Path {
        global: false,
        res: Res::Err,
        segments: vec![PathSegment {
            name,
            args: external_generic_args(cx, trait_did, has_self, bindings, substs),
        }],
    }
}

/// The point of this function is to replace bounds with types.
///
/// i.e. `[T, U]` when you have the following bounds: `T: Display, U: Option<T>` will return
/// `[Display, Option]` (we just returns the list of the types, we don't care about the
/// wrapped types in here).
crate fn get_real_types(
    generics: &Generics,
    arg: &Type,
    cx: &DocContext<'_>,
    recurse: i32,
) -> FxHashSet<(Type, TypeKind)> {
    let mut res = FxHashSet::default();
    if recurse >= 10 {
        // FIXME: remove this whole recurse thing when the recursion bug is fixed
        return res;
    }
    if arg.is_full_generic() {
        let arg_s = Symbol::intern(&arg.print().to_string());
        if let Some(where_pred) = generics.where_predicates.iter().find(|g| match g {
            &WherePredicate::BoundPredicate { ref ty, .. } => ty.def_id() == arg.def_id(),
            _ => false,
        }) {
            let bounds = where_pred.get_bounds().unwrap_or_else(|| &[]);
            for bound in bounds.iter() {
                if let GenericBound::TraitBound(ref poly_trait, _) = *bound {
                    for x in poly_trait.generic_params.iter() {
                        if !x.is_type() {
                            continue;
                        }
                        if let Some(ty) = x.get_type() {
                            let adds = get_real_types(generics, &ty, cx, recurse + 1);
                            if !adds.is_empty() {
                                res.extend(adds);
                            } else if !ty.is_full_generic() {
                                if let Some(kind) =
                                    ty.def_id().map(|did| cx.tcx.def_kind(did).clean(cx))
                                {
                                    res.insert((ty, kind));
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(bound) = generics.params.iter().find(|g| g.is_type() && g.name == arg_s) {
            for bound in bound.get_bounds().unwrap_or_else(|| &[]) {
                if let Some(ty) = bound.get_trait_type() {
                    let adds = get_real_types(generics, &ty, cx, recurse + 1);
                    if !adds.is_empty() {
                        res.extend(adds);
                    } else if !ty.is_full_generic() {
                        if let Some(kind) = ty.def_id().map(|did| cx.tcx.def_kind(did).clean(cx)) {
                            res.insert((ty.clone(), kind));
                        }
                    }
                }
            }
        }
    } else {
        if let Some(kind) = arg.def_id().map(|did| cx.tcx.def_kind(did).clean(cx)) {
            res.insert((arg.clone(), kind));
        }
        if let Some(gens) = arg.generics() {
            for gen in gens.iter() {
                if gen.is_full_generic() {
                    let adds = get_real_types(generics, gen, cx, recurse + 1);
                    if !adds.is_empty() {
                        res.extend(adds);
                    }
                } else if let Some(kind) = gen.def_id().map(|did| cx.tcx.def_kind(did).clean(cx)) {
                    res.insert((gen.clone(), kind));
                }
            }
        }
    }
    res
}

/// Return the full list of types when bounds have been resolved.
///
/// i.e. `fn foo<A: Display, B: Option<A>>(x: u32, y: B)` will return
/// `[u32, Display, Option]`.
crate fn get_all_types(
    generics: &Generics,
    decl: &FnDecl,
    cx: &DocContext<'_>,
) -> (Vec<(Type, TypeKind)>, Vec<(Type, TypeKind)>) {
    let mut all_types = FxHashSet::default();
    for arg in decl.inputs.values.iter() {
        if arg.type_.is_self_type() {
            continue;
        }
        let args = get_real_types(generics, &arg.type_, cx, 0);
        if !args.is_empty() {
            all_types.extend(args);
        } else {
            if let Some(kind) = arg.type_.def_id().map(|did| cx.tcx.def_kind(did).clean(cx)) {
                all_types.insert((arg.type_.clone(), kind));
            }
        }
    }

    let ret_types = match decl.output {
        FnRetTy::Return(ref return_type) => {
            let mut ret = get_real_types(generics, &return_type, cx, 0);
            if ret.is_empty() {
                if let Some(kind) = return_type.def_id().map(|did| cx.tcx.def_kind(did).clean(cx)) {
                    ret.insert((return_type.clone(), kind));
                }
            }
            ret.into_iter().collect()
        }
        _ => Vec::new(),
    };
    (all_types.into_iter().collect(), ret_types)
}

crate fn strip_type(ty: Type) -> Type {
    match ty {
        Type::ResolvedPath { path, param_names, did, is_generic } => {
            Type::ResolvedPath { path: strip_path(&path), param_names, did, is_generic }
        }
        Type::Tuple(inner_tys) => {
            Type::Tuple(inner_tys.iter().map(|t| strip_type(t.clone())).collect())
        }
        Type::Slice(inner_ty) => Type::Slice(Box::new(strip_type(*inner_ty))),
        Type::Array(inner_ty, s) => Type::Array(Box::new(strip_type(*inner_ty)), s),
        Type::RawPointer(m, inner_ty) => Type::RawPointer(m, Box::new(strip_type(*inner_ty))),
        Type::BorrowedRef { lifetime, mutability, type_ } => {
            Type::BorrowedRef { lifetime, mutability, type_: Box::new(strip_type(*type_)) }
        }
        Type::QPath { name, self_type, trait_ } => Type::QPath {
            name,
            self_type: Box::new(strip_type(*self_type)),
            trait_: Box::new(strip_type(*trait_)),
        },
        _ => ty,
    }
}

crate fn strip_path(path: &Path) -> Path {
    let segments = path
        .segments
        .iter()
        .map(|s| PathSegment {
            name: s.name.clone(),
            args: GenericArgs::AngleBracketed { args: vec![], bindings: vec![] },
        })
        .collect();

    Path { global: path.global, res: path.res, segments }
}

crate fn qpath_to_string(p: &hir::QPath<'_>) -> String {
    let segments = match *p {
        hir::QPath::Resolved(_, ref path) => &path.segments,
        hir::QPath::TypeRelative(_, ref segment) => return segment.ident.to_string(),
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

crate fn build_deref_target_impls(cx: &DocContext<'_>, items: &[Item], ret: &mut Vec<Item>) {
    let tcx = cx.tcx;

    for item in items {
        let target = match *item.kind {
            ItemKind::TypedefItem(ref t, true) => &t.type_,
            _ => continue,
        };
        let primitive = match *target {
            ResolvedPath { did, .. } if did.is_local() => continue,
            ResolvedPath { did, .. } => {
                ret.extend(inline::build_impls(cx, None, did, None));
                continue;
            }
            _ => match target.primitive_type() {
                Some(prim) => prim,
                None => continue,
            },
        };
        for &did in primitive.impls(tcx) {
            if !did.is_local() {
                inline::build_impl(cx, None, did, None, ret);
            }
        }
    }
}

crate trait ToSource {
    fn to_src(&self, cx: &DocContext<'_>) -> String;
}

impl ToSource for rustc_span::Span {
    fn to_src(&self, cx: &DocContext<'_>) -> String {
        debug!("converting span {:?} to snippet", self.clean(cx));
        let sn = match cx.sess().source_map().span_to_snippet(*self) {
            Ok(x) => x,
            Err(_) => String::new(),
        };
        debug!("got snippet {}", sn);
        sn
    }
}

crate fn name_from_pat(p: &hir::Pat<'_>) -> Symbol {
    use rustc_hir::*;
    debug!("trying to get a name from pattern: {:?}", p);

    Symbol::intern(&match p.kind {
        PatKind::Wild => return kw::Underscore,
        PatKind::Binding(_, _, ident, _) => return ident.name,
        PatKind::TupleStruct(ref p, ..) | PatKind::Path(ref p) => qpath_to_string(p),
        PatKind::Struct(ref name, ref fields, etc) => format!(
            "{} {{ {}{} }}",
            qpath_to_string(name),
            fields
                .iter()
                .map(|fp| format!("{}: {}", fp.ident, name_from_pat(&fp.pat)))
                .collect::<Vec<String>>()
                .join(", "),
            if etc { ", .." } else { "" }
        ),
        PatKind::Or(ref pats) => pats
            .iter()
            .map(|p| name_from_pat(&**p).to_string())
            .collect::<Vec<String>>()
            .join(" | "),
        PatKind::Tuple(ref elts, _) => format!(
            "({})",
            elts.iter()
                .map(|p| name_from_pat(&**p).to_string())
                .collect::<Vec<String>>()
                .join(", ")
        ),
        PatKind::Box(ref p) => return name_from_pat(&**p),
        PatKind::Ref(ref p, _) => return name_from_pat(&**p),
        PatKind::Lit(..) => {
            warn!(
                "tried to get argument name from PatKind::Lit, which is silly in function arguments"
            );
            return Symbol::intern("()");
        }
        PatKind::Range(..) => panic!(
            "tried to get argument name from PatKind::Range, \
             which is not allowed in function arguments"
        ),
        PatKind::Slice(ref begin, ref mid, ref end) => {
            let begin = begin.iter().map(|p| name_from_pat(&**p).to_string());
            let mid = mid.as_ref().map(|p| format!("..{}", name_from_pat(&**p))).into_iter();
            let end = end.iter().map(|p| name_from_pat(&**p).to_string());
            format!("[{}]", begin.chain(mid).chain(end).collect::<Vec<_>>().join(", "))
        }
    })
}

crate fn print_const(cx: &DocContext<'_>, n: &'tcx ty::Const<'_>) -> String {
    match n.val {
        ty::ConstKind::Unevaluated(def, _, promoted) => {
            let mut s = if let Some(def) = def.as_local() {
                let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def.did);
                print_const_expr(cx, cx.tcx.hir().body_owned_by(hir_id))
            } else {
                inline::print_inlined_const(cx, def.did)
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

crate fn print_evaluated_const(cx: &DocContext<'_>, def_id: DefId) -> Option<String> {
    cx.tcx.const_eval_poly(def_id).ok().and_then(|val| {
        let ty = cx.tcx.type_of(def_id);
        match (val, ty.kind()) {
            (_, &ty::Ref(..)) => None,
            (ConstValue::Scalar(_), &ty::Adt(_, _)) => None,
            (ConstValue::Scalar(_), _) => {
                let const_ = ty::Const::from_value(cx.tcx, val, ty);
                Some(print_const_with_custom_print_scalar(cx, const_))
            }
            _ => None,
        }
    })
}

fn format_integer_with_underscore_sep(num: &str) -> String {
    let num_chars: Vec<_> = num.chars().collect();
    let num_start_index = if num_chars.get(0) == Some(&'-') { 1 } else { 0 };

    num_chars[..num_start_index]
        .iter()
        .chain(num_chars[num_start_index..].rchunks(3).rev().intersperse(&['_']).flatten())
        .collect()
}

fn print_const_with_custom_print_scalar(cx: &DocContext<'_>, ct: &'tcx ty::Const<'tcx>) -> String {
    // Use a slightly different format for integer types which always shows the actual value.
    // For all other types, fallback to the original `pretty_print_const`.
    match (ct.val, ct.ty.kind()) {
        (ty::ConstKind::Value(ConstValue::Scalar(int)), ty::Uint(ui)) => {
            format!("{}{}", format_integer_with_underscore_sep(&int.to_string()), ui.name_str())
        }
        (ty::ConstKind::Value(ConstValue::Scalar(int)), ty::Int(i)) => {
            let ty = cx.tcx.lift(ct.ty).unwrap();
            let size = cx.tcx.layout_of(ty::ParamEnv::empty().and(ty)).unwrap().size;
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

crate fn is_literal_expr(cx: &DocContext<'_>, hir_id: hir::HirId) -> bool {
    if let hir::Node::Expr(expr) = cx.tcx.hir().get(hir_id) {
        if let hir::ExprKind::Lit(_) = &expr.kind {
            return true;
        }

        if let hir::ExprKind::Unary(hir::UnOp::UnNeg, expr) = &expr.kind {
            if let hir::ExprKind::Lit(_) = &expr.kind {
                return true;
            }
        }
    }

    false
}

crate fn print_const_expr(cx: &DocContext<'_>, body: hir::BodyId) -> String {
    let value = &cx.tcx.hir().body(body).value;

    let snippet = if !value.span.from_expansion() {
        cx.sess().source_map().span_to_snippet(value.span).ok()
    } else {
        None
    };

    snippet.unwrap_or_else(|| rustc_hir_pretty::id_to_string(&cx.tcx.hir(), body.hir_id))
}

/// Given a type Path, resolve it to a Type using the TyCtxt
crate fn resolve_type(cx: &DocContext<'_>, path: Path, id: hir::HirId) -> Type {
    debug!("resolve_type({:?},{:?})", path, id);

    let is_generic = match path.res {
        Res::PrimTy(p) => return Primitive(PrimitiveType::from(p)),
        Res::SelfTy(..) if path.segments.len() == 1 => {
            return Generic(kw::SelfUpper);
        }
        Res::Def(DefKind::TyParam, _) if path.segments.len() == 1 => {
            return Generic(Symbol::intern(&format!("{:#}", path.print())));
        }
        Res::SelfTy(..) | Res::Def(DefKind::TyParam | DefKind::AssocTy, _) => true,
        _ => false,
    };
    let did = register_res(&*cx, path.res);
    ResolvedPath { path, param_names: None, did, is_generic }
}

crate fn get_auto_trait_and_blanket_impls(
    cx: &DocContext<'tcx>,
    ty: Ty<'tcx>,
    param_env_def_id: DefId,
) -> impl Iterator<Item = Item> {
    let auto_impls = cx.sess().time("get_auto_trait_impls", || {
        AutoTraitFinder::new(cx).get_auto_trait_impls(ty, param_env_def_id)
    });
    let blanket_impls = cx.sess().time("get_blanket_impls", || {
        BlanketImplFinder::new(cx).get_blanket_impls(ty, param_env_def_id)
    });
    auto_impls.into_iter().chain(blanket_impls)
}

crate fn register_res(cx: &DocContext<'_>, res: Res) -> DefId {
    debug!("register_res({:?})", res);

    let (did, kind) = match res {
        Res::Def(DefKind::Fn, i) => (i, TypeKind::Function),
        Res::Def(DefKind::TyAlias, i) => (i, TypeKind::Typedef),
        Res::Def(DefKind::Enum, i) => (i, TypeKind::Enum),
        Res::Def(DefKind::Trait, i) => (i, TypeKind::Trait),
        Res::Def(DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst, i) => {
            (cx.tcx.parent(i).unwrap(), TypeKind::Trait)
        }
        Res::Def(DefKind::Struct, i) => (i, TypeKind::Struct),
        Res::Def(DefKind::Union, i) => (i, TypeKind::Union),
        Res::Def(DefKind::Mod, i) => (i, TypeKind::Module),
        Res::Def(DefKind::ForeignTy, i) => (i, TypeKind::Foreign),
        Res::Def(DefKind::Const, i) => (i, TypeKind::Const),
        Res::Def(DefKind::Static, i) => (i, TypeKind::Static),
        Res::Def(DefKind::Variant, i) => {
            (cx.tcx.parent(i).expect("cannot get parent def id"), TypeKind::Enum)
        }
        Res::Def(DefKind::Macro(mac_kind), i) => match mac_kind {
            MacroKind::Bang => (i, TypeKind::Macro),
            MacroKind::Attr => (i, TypeKind::Attr),
            MacroKind::Derive => (i, TypeKind::Derive),
        },
        Res::Def(DefKind::TraitAlias, i) => (i, TypeKind::TraitAlias),
        Res::SelfTy(Some(def_id), _) => (def_id, TypeKind::Trait),
        Res::SelfTy(_, Some((impl_def_id, _))) => return impl_def_id,
        _ => return res.def_id(),
    };
    if did.is_local() {
        return did;
    }
    inline::record_extern_fqn(cx, did, kind);
    if let TypeKind::Trait = kind {
        inline::record_extern_trait(cx, did);
    }
    did
}

crate fn resolve_use_source(cx: &DocContext<'_>, path: Path) -> ImportSource {
    ImportSource {
        did: if path.res.opt_def_id().is_none() { None } else { Some(register_res(cx, path.res)) },
        path,
    }
}

crate fn enter_impl_trait<F, R>(cx: &DocContext<'_>, f: F) -> R
where
    F: FnOnce() -> R,
{
    let old_bounds = mem::take(&mut *cx.impl_trait_bounds.borrow_mut());
    let r = f();
    assert!(cx.impl_trait_bounds.borrow().is_empty());
    *cx.impl_trait_bounds.borrow_mut() = old_bounds;
    r
}
