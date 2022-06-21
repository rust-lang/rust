//! These from impls are used to create the JSON types which get serialized. They're very close to
//! the `clean` types but with some fields removed or stringified to simplify the output and not
//! expose unstable compiler internals.

#![allow(rustc::default_hash_types)]

use std::convert::From;
use std::fmt;
use std::rc::Rc;

use rustc_ast::ast;
use rustc_hir::{def::CtorKind, def_id::DefId};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{Pos, Symbol};
use rustc_target::spec::abi::Abi as RustcAbi;

use rustdoc_json_types::*;

use crate::clean::utils::print_const_expr;
use crate::clean::{self, ItemId};
use crate::formats::{cache::Cache, item_type::ItemType};
use crate::html::render::constant::Renderer as ConstantRenderer;
use crate::json::JsonRenderer;

impl JsonRenderer<'_> {
    pub(super) fn convert_item(&self, item: clean::Item) -> Option<Item> {
        let deprecation = item.deprecation(self.tcx);
        let links = self
            .cache
            .intra_doc_links
            .get(&item.item_id)
            .into_iter()
            .flatten()
            .map(|clean::ItemLink { link, did, .. }| {
                (link.clone(), from_item_id((*did).into(), self.tcx))
            })
            .collect();
        let docs = item.attrs.collapsed_doc_value();
        let attrs = item
            .attrs
            .other_attrs
            .iter()
            .map(rustc_ast_pretty::pprust::attribute_to_string)
            .collect();
        let span = item.span(self.tcx);
        let clean::Item { name, attrs: _, kind: _, visibility, item_id, cfg: _ } = item;
        let inner = match *item.kind {
            clean::KeywordItem => return None,
            clean::StrippedItem(ref inner) => {
                match &**inner {
                    // We document non-empty stripped modules as with `Module::is_stripped` set to
                    // `true`, to prevent contained items from being orphaned for downstream users,
                    // as JSON does no inlining.
                    clean::ModuleItem(m) if !m.items.is_empty() => {
                        from_clean_item(item, &Context::new(self.tcx, self.cache.clone()))
                    }
                    _ => return None,
                }
            }
            _ => from_clean_item(item, &Context::new(self.tcx, self.cache.clone())),
        };
        Some(Item {
            id: from_item_id_with_name(item_id, self.tcx, name),
            crate_id: item_id.krate().as_u32(),
            name: name.map(|sym| sym.to_string()),
            span: span.and_then(|span| self.convert_span(span)),
            visibility: self.convert_visibility(visibility),
            docs,
            attrs,
            deprecation: deprecation.map(from_deprecation),
            inner,
            links,
        })
    }

    fn convert_span(&self, span: clean::Span) -> Option<Span> {
        match span.filename(self.sess()) {
            rustc_span::FileName::Real(name) => {
                if let Some(local_path) = name.into_local_path() {
                    let hi = span.hi(self.sess());
                    let lo = span.lo(self.sess());
                    Some(Span {
                        filename: local_path,
                        begin: (lo.line, lo.col.to_usize()),
                        end: (hi.line, hi.col.to_usize()),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn convert_visibility(&self, v: clean::Visibility) -> Visibility {
        use clean::Visibility::*;
        match v {
            Public => Visibility::Public,
            Inherited => Visibility::Default,
            Restricted(did) if did.is_crate_root() => Visibility::Crate,
            Restricted(did) => Visibility::Restricted {
                parent: from_item_id(did.into(), self.tcx),
                path: self.tcx.def_path(did).to_string_no_crate_verbose(),
            },
        }
    }
}

#[derive(Clone)]
pub(crate) struct Context<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) cache: Rc<Cache>,
}

impl<'tcx> Context<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, cache: Rc<Cache>) -> Self {
        Self { tcx, cache }
    }
}

pub(crate) trait FromWithCx<T> {
    fn from_cx(f: T, cx: &Context<'_>) -> Self;
}

pub(crate) trait IntoWithCx<T> {
    fn into_cx(self, cx: &Context<'_>) -> T;
}

impl<T, U> IntoWithCx<U> for T
where
    U: FromWithCx<T>,
{
    fn into_cx(self, cx: &Context<'_>) -> U {
        U::from_cx(self, cx)
    }
}

impl<I, T, U> FromWithCx<I> for Vec<U>
where
    I: IntoIterator<Item = T>,
    U: FromWithCx<T>,
{
    fn from_cx(f: I, cx: &Context<'_>) -> Vec<U> {
        f.into_iter().map(|x| x.into_cx(cx)).collect()
    }
}

pub(crate) fn from_deprecation(deprecation: rustc_attr::Deprecation) -> Deprecation {
    #[rustfmt::skip]
    let rustc_attr::Deprecation { since, note, is_since_rustc_version: _, suggestion: _ } = deprecation;
    Deprecation { since: since.map(|s| s.to_string()), note: note.map(|s| s.to_string()) }
}

impl FromWithCx<clean::GenericArgs> for GenericArgs {
    fn from_cx(args: clean::GenericArgs, cx: &Context<'_>) -> Self {
        use clean::GenericArgs::*;
        match args {
            AngleBracketed { args, bindings } => GenericArgs::AngleBracketed {
                args: args.into_vec().into_cx(cx),
                bindings: bindings.into_cx(cx),
            },
            Parenthesized { inputs, output } => GenericArgs::Parenthesized {
                inputs: inputs.into_vec().into_cx(cx),
                output: output.map(|a| (*a).into_cx(cx)),
            },
        }
    }
}

impl FromWithCx<clean::GenericArg> for GenericArg {
    fn from_cx(arg: clean::GenericArg, cx: &Context<'_>) -> Self {
        use clean::GenericArg::*;
        match arg {
            Lifetime(l) => GenericArg::Lifetime(convert_lifetime(l)),
            Type(t) => GenericArg::Type(t.into_cx(cx)),
            Const(box c) => GenericArg::Const(c.into_cx(cx)),
            Infer => GenericArg::Infer,
        }
    }
}

impl FromWithCx<clean::Constant> for Constant {
    fn from_cx(constant: clean::Constant, cx: &Context<'_>) -> Self {
        let expr = constant.expr(cx.tcx);
        // FIXME: Should we “disable” depth and length limits for the JSON backend?
        let value = constant.eval_and_render(&ConstantRenderer::PlainText(cx.clone()));
        let is_literal = constant.is_literal(cx.tcx);
        Constant { type_: constant.type_.into_cx(cx), expr, value, is_literal }
    }
}

impl FromWithCx<clean::TypeBinding> for TypeBinding {
    fn from_cx(binding: clean::TypeBinding, cx: &Context<'_>) -> Self {
        TypeBinding {
            name: binding.assoc.name.to_string(),
            args: binding.assoc.args.into_cx(cx),
            binding: binding.kind.into_cx(cx),
        }
    }
}

impl FromWithCx<clean::TypeBindingKind> for TypeBindingKind {
    fn from_cx(kind: clean::TypeBindingKind, cx: &Context<'_>) -> Self {
        use clean::TypeBindingKind::*;
        match kind {
            Equality { term } => TypeBindingKind::Equality(term.into_cx(cx)),
            Constraint { bounds } => TypeBindingKind::Constraint(bounds.into_cx(cx)),
        }
    }
}

/// It generates an ID as follows:
///
/// `CRATE_ID:ITEM_ID[:NAME_ID]` (if there is no name, NAME_ID is not generated).
pub(crate) fn from_item_id(item_id: ItemId, tcx: TyCtxt<'_>) -> Id {
    from_item_id_with_name(item_id, tcx, None)
}

// FIXME: this function (and appending the name at the end of the ID) should be removed when
// reexports are not inlined anymore for json format. It should be done in #93518.
pub(crate) fn from_item_id_with_name(item_id: ItemId, tcx: TyCtxt<'_>, name: Option<Symbol>) -> Id {
    struct DisplayDefId<'a>(DefId, TyCtxt<'a>, Option<Symbol>);

    impl<'a> fmt::Display for DisplayDefId<'a> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let name = match self.2 {
                Some(name) => format!(":{}", name.as_u32()),
                None => self
                    .1
                    .opt_item_name(self.0)
                    .map(|n| format!(":{}", n.as_u32()))
                    .unwrap_or_default(),
            };
            write!(f, "{}:{}{}", self.0.krate.as_u32(), u32::from(self.0.index), name)
        }
    }

    match item_id {
        ItemId::DefId(did) => Id(format!("{}", DisplayDefId(did, tcx, name))),
        ItemId::Blanket { for_, impl_id } => {
            Id(format!("b:{}-{}", DisplayDefId(impl_id, tcx, None), DisplayDefId(for_, tcx, name)))
        }
        ItemId::Auto { for_, trait_ } => {
            Id(format!("a:{}-{}", DisplayDefId(trait_, tcx, None), DisplayDefId(for_, tcx, name)))
        }
        ItemId::Primitive(ty, krate) => Id(format!("p:{}:{}", krate.as_u32(), ty.as_sym())),
    }
}

fn from_clean_item(item: clean::Item, cx: &Context<'_>) -> ItemEnum {
    use clean::ItemKind::*;
    let name = item.name;
    let is_crate = item.is_crate();
    let header = item.fn_header(cx.tcx);

    match *item.kind {
        ModuleItem(m) => {
            ItemEnum::Module(Module { is_crate, items: ids(m.items, cx.tcx), is_stripped: false })
        }
        ImportItem(i) => ItemEnum::Import(i.into_cx(cx)),
        StructItem(s) => ItemEnum::Struct(s.into_cx(cx)),
        UnionItem(u) => ItemEnum::Union(u.into_cx(cx)),
        StructFieldItem(f) => ItemEnum::StructField(f.into_cx(cx)),
        EnumItem(e) => ItemEnum::Enum(e.into_cx(cx)),
        VariantItem(v) => ItemEnum::Variant(v.into_cx(cx)),
        FunctionItem(f) => ItemEnum::Function(from_function(f, header.unwrap(), cx)),
        ForeignFunctionItem(f) => ItemEnum::Function(from_function(f, header.unwrap(), cx)),
        TraitItem(t) => ItemEnum::Trait(t.into_cx(cx)),
        TraitAliasItem(t) => ItemEnum::TraitAlias(t.into_cx(cx)),
        MethodItem(m, _) => ItemEnum::Method(from_function_method(m, true, header.unwrap(), cx)),
        TyMethodItem(m) => ItemEnum::Method(from_function_method(m, false, header.unwrap(), cx)),
        ImplItem(i) => ItemEnum::Impl((*i).into_cx(cx)),
        StaticItem(s) => ItemEnum::Static(s.into_cx(cx)),
        ForeignStaticItem(s) => ItemEnum::Static(s.into_cx(cx)),
        ForeignTypeItem => ItemEnum::ForeignType,
        TypedefItem(t) => ItemEnum::Typedef(t.into_cx(cx)),
        OpaqueTyItem(t) => ItemEnum::OpaqueTy(t.into_cx(cx)),
        ConstantItem(c) => ItemEnum::Constant(c.into_cx(cx)),
        MacroItem(m) => ItemEnum::Macro(m.source),
        ProcMacroItem(m) => ItemEnum::ProcMacro(m.into_cx(cx)),
        PrimitiveItem(p) => ItemEnum::PrimitiveType(p.as_sym().to_string()),
        TyAssocConstItem(ty) => ItemEnum::AssocConst { type_: ty.into_cx(cx), default: None },
        AssocConstItem(ty, default) => ItemEnum::AssocConst {
            type_: ty.into_cx(cx),
            // FIXME: Should we “disable” depth and length limits for the JSON backend?
            default: Some(
                default
                    .eval_and_render(&ConstantRenderer::PlainText(cx.clone()))
                    .unwrap_or_else(|| default.expr(cx.tcx)),
            ),
        },
        TyAssocTypeItem(g, b) => {
            ItemEnum::AssocType { generics: (*g).into_cx(cx), bounds: b.into_cx(cx), default: None }
        }
        AssocTypeItem(t, b) => ItemEnum::AssocType {
            generics: t.generics.into_cx(cx),
            bounds: b.into_cx(cx),
            default: Some(t.item_type.unwrap_or(t.type_).into_cx(cx)),
        },
        // `convert_item` early returns `None` for stripped items and keywords.
        KeywordItem => unreachable!(),
        StrippedItem(inner) => {
            match *inner {
                ModuleItem(m) => ItemEnum::Module(Module {
                    is_crate,
                    items: ids(m.items, cx.tcx),
                    is_stripped: true,
                }),
                // `convert_item` early returns `None` for stripped items we're not including
                _ => unreachable!(),
            }
        }
        ExternCrateItem { ref src } => ItemEnum::ExternCrate {
            name: name.as_ref().unwrap().to_string(),
            rename: src.map(|x| x.to_string()),
        },
    }
}

impl FromWithCx<clean::Struct> for Struct {
    fn from_cx(struct_: clean::Struct, cx: &Context<'_>) -> Self {
        let fields_stripped = struct_.has_stripped_entries();
        let clean::Struct { struct_type, generics, fields } = struct_;
        Struct {
            struct_type: from_ctor_kind(struct_type),
            generics: generics.into_cx(cx),
            fields_stripped,
            fields: ids(fields, cx.tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithCx<clean::Union> for Union {
    fn from_cx(union_: clean::Union, cx: &Context<'_>) -> Self {
        let fields_stripped = union_.has_stripped_entries();
        let clean::Union { generics, fields } = union_;
        Union {
            generics: generics.into_cx(cx),
            fields_stripped,
            fields: ids(fields, cx.tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

pub(crate) fn from_ctor_kind(struct_type: CtorKind) -> StructType {
    match struct_type {
        CtorKind::Fictive => StructType::Plain,
        CtorKind::Fn => StructType::Tuple,
        CtorKind::Const => StructType::Unit,
    }
}

pub(crate) fn from_fn_header(header: &rustc_hir::FnHeader) -> Header {
    Header {
        async_: header.is_async(),
        const_: header.is_const(),
        unsafe_: header.is_unsafe(),
        abi: convert_abi(header.abi),
    }
}

fn convert_abi(a: RustcAbi) -> Abi {
    match a {
        RustcAbi::Rust => Abi::Rust,
        RustcAbi::C { unwind } => Abi::C { unwind },
        RustcAbi::Cdecl { unwind } => Abi::Cdecl { unwind },
        RustcAbi::Stdcall { unwind } => Abi::Stdcall { unwind },
        RustcAbi::Fastcall { unwind } => Abi::Fastcall { unwind },
        RustcAbi::Aapcs { unwind } => Abi::Aapcs { unwind },
        RustcAbi::Win64 { unwind } => Abi::Win64 { unwind },
        RustcAbi::SysV64 { unwind } => Abi::SysV64 { unwind },
        RustcAbi::System { unwind } => Abi::System { unwind },
        _ => Abi::Other(a.to_string()),
    }
}

fn convert_lifetime(l: clean::Lifetime) -> String {
    l.0.to_string()
}

impl FromWithCx<clean::Generics> for Generics {
    fn from_cx(generics: clean::Generics, cx: &Context<'_>) -> Self {
        Generics {
            params: generics.params.into_cx(cx),
            where_predicates: generics.where_predicates.into_cx(cx),
        }
    }
}

impl FromWithCx<clean::GenericParamDef> for GenericParamDef {
    fn from_cx(generic_param: clean::GenericParamDef, cx: &Context<'_>) -> Self {
        GenericParamDef {
            name: generic_param.name.to_string(),
            kind: generic_param.kind.into_cx(cx),
        }
    }
}

impl FromWithCx<clean::GenericParamDefKind> for GenericParamDefKind {
    fn from_cx(kind: clean::GenericParamDefKind, cx: &Context<'_>) -> Self {
        use clean::GenericParamDefKind::*;
        match kind {
            Lifetime { outlives } => GenericParamDefKind::Lifetime {
                outlives: outlives.into_iter().map(convert_lifetime).collect(),
            },
            Type { did: _, bounds, default, synthetic } => GenericParamDefKind::Type {
                bounds: bounds.into_cx(cx),
                default: default.map(|x| (*x).into_cx(cx)),
                synthetic,
            },
            Const { did: _, ty, default } => GenericParamDefKind::Const {
                type_: (*ty).into_cx(cx),
                default: default.map(|x| *x),
            },
        }
    }
}

impl FromWithCx<clean::WherePredicate> for WherePredicate {
    fn from_cx(predicate: clean::WherePredicate, cx: &Context<'_>) -> Self {
        use clean::WherePredicate::*;
        match predicate {
            BoundPredicate { ty, bounds, bound_params } => WherePredicate::BoundPredicate {
                type_: ty.into_cx(cx),
                bounds: bounds.into_cx(cx),
                generic_params: bound_params
                    .into_iter()
                    .map(|x| GenericParamDef {
                        name: x.0.to_string(),
                        kind: GenericParamDefKind::Lifetime { outlives: vec![] },
                    })
                    .collect(),
            },
            RegionPredicate { lifetime, bounds } => WherePredicate::RegionPredicate {
                lifetime: convert_lifetime(lifetime),
                bounds: bounds.into_cx(cx),
            },
            EqPredicate { lhs, rhs } => {
                WherePredicate::EqPredicate { lhs: lhs.into_cx(cx), rhs: rhs.into_cx(cx) }
            }
        }
    }
}

impl FromWithCx<clean::GenericBound> for GenericBound {
    fn from_cx(bound: clean::GenericBound, cx: &Context<'_>) -> Self {
        use clean::GenericBound::*;
        match bound {
            TraitBound(clean::PolyTrait { trait_, generic_params }, modifier) => {
                GenericBound::TraitBound {
                    trait_: trait_.into_cx(cx),
                    generic_params: generic_params.into_cx(cx),
                    modifier: from_trait_bound_modifier(modifier),
                }
            }
            Outlives(lifetime) => GenericBound::Outlives(convert_lifetime(lifetime)),
        }
    }
}

pub(crate) fn from_trait_bound_modifier(
    modifier: rustc_hir::TraitBoundModifier,
) -> TraitBoundModifier {
    use rustc_hir::TraitBoundModifier::*;
    match modifier {
        None => TraitBoundModifier::None,
        Maybe => TraitBoundModifier::Maybe,
        MaybeConst => TraitBoundModifier::MaybeConst,
    }
}

impl FromWithCx<clean::Type> for Type {
    fn from_cx(ty: clean::Type, cx: &Context<'_>) -> Self {
        use clean::Type::{
            Array, BareFunction, BorrowedRef, Generic, ImplTrait, Infer, Primitive, QPath,
            RawPointer, Slice, Tuple,
        };

        match ty {
            clean::Type::Path { path } => Type::ResolvedPath(path.into_cx(cx)),
            clean::Type::DynTrait(bounds, lt) => Type::DynTrait(DynTrait {
                lifetime: lt.map(convert_lifetime),
                traits: bounds.into_cx(cx),
            }),
            Generic(s) => Type::Generic(s.to_string()),
            Primitive(p) => Type::Primitive(p.as_sym().to_string()),
            BareFunction(f) => Type::FunctionPointer(Box::new((*f).into_cx(cx))),
            Tuple(t) => Type::Tuple(t.into_cx(cx)),
            Slice(t) => Type::Slice(Box::new((*t).into_cx(cx))),
            Array(t, s) => Type::Array { type_: Box::new((*t).into_cx(cx)), len: s },
            ImplTrait(g) => Type::ImplTrait(g.into_cx(cx)),
            Infer => Type::Infer,
            RawPointer(mutability, type_) => Type::RawPointer {
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_cx(cx)),
            },
            BorrowedRef { lifetime, mutability, type_ } => Type::BorrowedRef {
                lifetime: lifetime.map(convert_lifetime),
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_cx(cx)),
            },
            QPath { assoc, self_type, trait_, .. } => Type::QualifiedPath {
                name: assoc.name.to_string(),
                args: Box::new(assoc.args.clone().into_cx(cx)),
                self_type: Box::new((*self_type).into_cx(cx)),
                trait_: trait_.into_cx(cx),
            },
        }
    }
}

impl FromWithCx<clean::Path> for Path {
    fn from_cx(path: clean::Path, cx: &Context<'_>) -> Path {
        Path {
            name: path.whole_name(),
            id: from_item_id(path.def_id().into(), cx.tcx),
            args: path.segments.last().map(|args| Box::new(args.clone().args.into_cx(cx))),
        }
    }
}

impl FromWithCx<clean::Term> for Term {
    fn from_cx(term: clean::Term, cx: &Context<'_>) -> Term {
        match term {
            clean::Term::Type(ty) => Term::Type(FromWithCx::from_cx(ty, cx)),
            clean::Term::Constant(c) => Term::Constant(FromWithCx::from_cx(c, cx)),
        }
    }
}

impl FromWithCx<clean::BareFunctionDecl> for FunctionPointer {
    fn from_cx(bare_decl: clean::BareFunctionDecl, cx: &Context<'_>) -> Self {
        let clean::BareFunctionDecl { unsafety, generic_params, decl, abi } = bare_decl;
        FunctionPointer {
            header: Header {
                unsafe_: matches!(unsafety, rustc_hir::Unsafety::Unsafe),
                const_: false,
                async_: false,
                abi: convert_abi(abi),
            },
            generic_params: generic_params.into_cx(cx),
            decl: decl.into_cx(cx),
        }
    }
}

impl FromWithCx<clean::FnDecl> for FnDecl {
    fn from_cx(decl: clean::FnDecl, cx: &Context<'_>) -> Self {
        let clean::FnDecl { inputs, output, c_variadic } = decl;
        FnDecl {
            inputs: inputs
                .values
                .into_iter()
                .map(|arg| (arg.name.to_string(), arg.type_.into_cx(cx)))
                .collect(),
            output: match output {
                clean::FnRetTy::Return(t) => Some(t.into_cx(cx)),
                clean::FnRetTy::DefaultReturn => None,
            },
            c_variadic,
        }
    }
}

impl FromWithCx<clean::Trait> for Trait {
    fn from_cx(trait_: clean::Trait, cx: &Context<'_>) -> Self {
        let is_auto = trait_.is_auto(cx.tcx);
        let is_unsafe = trait_.unsafety(cx.tcx) == rustc_hir::Unsafety::Unsafe;
        let clean::Trait { items, generics, bounds, .. } = trait_;
        Trait {
            is_auto,
            is_unsafe,
            items: ids(items, cx.tcx),
            generics: generics.into_cx(cx),
            bounds: bounds.into_cx(cx),
            implementations: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithCx<clean::PolyTrait> for PolyTrait {
    fn from_cx(
        clean::PolyTrait { trait_, generic_params }: clean::PolyTrait,
        cx: &Context<'_>,
    ) -> Self {
        PolyTrait { trait_: trait_.into_cx(cx), generic_params: generic_params.into_cx(cx) }
    }
}

impl FromWithCx<clean::Impl> for Impl {
    fn from_cx(impl_: clean::Impl, cx: &Context<'_>) -> Self {
        let provided_trait_methods = impl_.provided_trait_methods(cx.tcx);
        let clean::Impl { unsafety, generics, trait_, for_, items, polarity, kind } = impl_;
        // FIXME: use something like ImplKind in JSON?
        let (synthetic, blanket_impl) = match kind {
            clean::ImplKind::Normal | clean::ImplKind::FakeVaradic => (false, None),
            clean::ImplKind::Auto => (true, None),
            clean::ImplKind::Blanket(ty) => (false, Some(*ty)),
        };
        let negative_polarity = match polarity {
            ty::ImplPolarity::Positive | ty::ImplPolarity::Reservation => false,
            ty::ImplPolarity::Negative => true,
        };
        Impl {
            is_unsafe: unsafety == rustc_hir::Unsafety::Unsafe,
            generics: generics.into_cx(cx),
            provided_trait_methods: provided_trait_methods
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            trait_: trait_.map(|path| path.into_cx(cx)),
            for_: for_.into_cx(cx),
            items: ids(items, cx.tcx),
            negative: negative_polarity,
            synthetic,
            blanket_impl: blanket_impl.map(|x| x.into_cx(cx)),
        }
    }
}

pub(crate) fn from_function(
    function: Box<clean::Function>,
    header: rustc_hir::FnHeader,
    cx: &Context<'_>,
) -> Function {
    let clean::Function { decl, generics } = *function;
    Function {
        decl: decl.into_cx(cx),
        generics: generics.into_cx(cx),
        header: from_fn_header(&header),
    }
}

pub(crate) fn from_function_method(
    function: Box<clean::Function>,
    has_body: bool,
    header: rustc_hir::FnHeader,
    cx: &Context<'_>,
) -> Method {
    let clean::Function { decl, generics } = *function;
    Method {
        decl: decl.into_cx(cx),
        generics: generics.into_cx(cx),
        header: from_fn_header(&header),
        has_body,
    }
}

impl FromWithCx<clean::Enum> for Enum {
    fn from_cx(enum_: clean::Enum, cx: &Context<'_>) -> Self {
        let variants_stripped = enum_.has_stripped_entries();
        let clean::Enum { variants, generics } = enum_;
        Enum {
            generics: generics.into_cx(cx),
            variants_stripped,
            variants: ids(variants, cx.tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithCx<clean::VariantStruct> for Struct {
    fn from_cx(struct_: clean::VariantStruct, cx: &Context<'_>) -> Self {
        let fields_stripped = struct_.has_stripped_entries();
        let clean::VariantStruct { struct_type, fields } = struct_;
        Struct {
            struct_type: from_ctor_kind(struct_type),
            generics: Generics { params: vec![], where_predicates: vec![] },
            fields_stripped,
            fields: ids(fields, cx.tcx),
            impls: Vec::new(),
        }
    }
}

impl FromWithCx<clean::Variant> for Variant {
    fn from_cx(variant: clean::Variant, cx: &Context<'_>) -> Self {
        use clean::Variant::*;
        match variant {
            CLike => Variant::Plain,
            Tuple(fields) => Variant::Tuple(
                fields
                    .into_iter()
                    .map(|f| {
                        if let clean::StructFieldItem(ty) = *f.kind {
                            ty.into_cx(cx)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect(),
            ),
            Struct(s) => Variant::Struct(ids(s.fields, cx.tcx)),
        }
    }
}

impl FromWithCx<clean::Import> for Import {
    fn from_cx(import: clean::Import, cx: &Context<'_>) -> Self {
        use clean::ImportKind::*;
        let (name, glob) = match import.kind {
            Simple(s) => (s.to_string(), false),
            Glob => (
                import.source.path.last_opt().unwrap_or_else(|| Symbol::intern("*")).to_string(),
                true,
            ),
        };
        Import {
            source: import.source.path.whole_name(),
            name,
            id: import.source.did.map(ItemId::from).map(|i| from_item_id(i, cx.tcx)),
            glob,
        }
    }
}

impl FromWithCx<clean::ProcMacro> for ProcMacro {
    fn from_cx(mac: clean::ProcMacro, _cx: &Context<'_>) -> Self {
        ProcMacro {
            kind: from_macro_kind(mac.kind),
            helpers: mac.helpers.iter().map(|x| x.to_string()).collect(),
        }
    }
}

pub(crate) fn from_macro_kind(kind: rustc_span::hygiene::MacroKind) -> MacroKind {
    use rustc_span::hygiene::MacroKind::*;
    match kind {
        Bang => MacroKind::Bang,
        Attr => MacroKind::Attr,
        Derive => MacroKind::Derive,
    }
}

impl FromWithCx<Box<clean::Typedef>> for Typedef {
    fn from_cx(typedef: Box<clean::Typedef>, cx: &Context<'_>) -> Self {
        let clean::Typedef { type_, generics, item_type: _ } = *typedef;
        Typedef { type_: type_.into_cx(cx), generics: generics.into_cx(cx) }
    }
}

impl FromWithCx<clean::OpaqueTy> for OpaqueTy {
    fn from_cx(opaque: clean::OpaqueTy, cx: &Context<'_>) -> Self {
        OpaqueTy { bounds: opaque.bounds.into_cx(cx), generics: opaque.generics.into_cx(cx) }
    }
}

impl FromWithCx<clean::Static> for Static {
    fn from_cx(stat: clean::Static, cx: &Context<'_>) -> Self {
        Static {
            type_: stat.type_.into_cx(cx),
            mutable: stat.mutability == ast::Mutability::Mut,
            // FIXME: Consider using `eval_and_render` here instead (despite the name of the field)
            expr: stat.expr.map(|e| print_const_expr(cx.tcx, e)).unwrap_or_default(),
        }
    }
}

impl FromWithCx<clean::TraitAlias> for TraitAlias {
    fn from_cx(alias: clean::TraitAlias, cx: &Context<'_>) -> Self {
        TraitAlias { generics: alias.generics.into_cx(cx), params: alias.bounds.into_cx(cx) }
    }
}

impl FromWithCx<ItemType> for ItemKind {
    fn from_cx(kind: ItemType, _cx: &Context<'_>) -> Self {
        use ItemType::*;
        match kind {
            Module => ItemKind::Module,
            ExternCrate => ItemKind::ExternCrate,
            Import => ItemKind::Import,
            Struct => ItemKind::Struct,
            Union => ItemKind::Union,
            Enum => ItemKind::Enum,
            Function => ItemKind::Function,
            Typedef => ItemKind::Typedef,
            OpaqueTy => ItemKind::OpaqueTy,
            Static => ItemKind::Static,
            Constant => ItemKind::Constant,
            Trait => ItemKind::Trait,
            Impl => ItemKind::Impl,
            TyMethod | Method => ItemKind::Method,
            StructField => ItemKind::StructField,
            Variant => ItemKind::Variant,
            Macro => ItemKind::Macro,
            Primitive => ItemKind::Primitive,
            AssocConst => ItemKind::AssocConst,
            AssocType => ItemKind::AssocType,
            ForeignType => ItemKind::ForeignType,
            Keyword => ItemKind::Keyword,
            TraitAlias => ItemKind::TraitAlias,
            ProcAttribute => ItemKind::ProcAttribute,
            ProcDerive => ItemKind::ProcDerive,
        }
    }
}

fn ids(items: impl IntoIterator<Item = clean::Item>, tcx: TyCtxt<'_>) -> Vec<Id> {
    items
        .into_iter()
        .filter(|x| !x.is_stripped() && !x.is_keyword())
        .map(|i| from_item_id_with_name(i.item_id, tcx, i.name))
        .collect()
}
