//! These from impls are used to create the JSON types which get serialized. They're very close to
//! the `clean` types but with some fields removed or stringified to simplify the output and not
//! expose unstable compiler internals.

#![allow(rustc::default_hash_types)]

use std::convert::From;
use std::fmt;

use rustc_ast::ast;
use rustc_hir::{def::CtorKind, def_id::DefId};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{Pos, Symbol};
use rustc_target::spec::abi::Abi as RustcAbi;

use rustdoc_json_types::*;

use crate::clean::utils::print_const_expr;
use crate::clean::{self, ItemId};
use crate::formats::item_type::ItemType;
use crate::json::JsonRenderer;
use crate::passes::collect_intra_doc_links::UrlFragment;

impl JsonRenderer<'_> {
    pub(super) fn convert_item(&self, item: clean::Item) -> Option<Item> {
        let deprecation = item.deprecation(self.tcx);
        let links = self
            .cache
            .intra_doc_links
            .get(&item.item_id)
            .into_iter()
            .flatten()
            .map(|clean::ItemLink { link, page_id, fragment, .. }| {
                let id = match fragment {
                    Some(UrlFragment::Item(frag_id)) => *frag_id,
                    // FIXME: Pass the `UserWritten` segment to JSON consumer.
                    Some(UrlFragment::UserWritten(_)) | None => *page_id,
                };

                (link.clone(), from_item_id(id.into(), self.tcx))
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
                    // We document stripped modules as with `Module::is_stripped` set to
                    // `true`, to prevent contained items from being orphaned for downstream users,
                    // as JSON does no inlining.
                    clean::ModuleItem(_)
                        if self.imported_items.contains(&item_id.expect_def_id()) =>
                    {
                        from_clean_item(item, self.tcx)
                    }
                    _ => return None,
                }
            }
            _ => from_clean_item(item, self.tcx),
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

pub(crate) trait FromWithTcx<T> {
    fn from_tcx(f: T, tcx: TyCtxt<'_>) -> Self;
}

pub(crate) trait IntoWithTcx<T> {
    fn into_tcx(self, tcx: TyCtxt<'_>) -> T;
}

impl<T, U> IntoWithTcx<U> for T
where
    U: FromWithTcx<T>,
{
    fn into_tcx(self, tcx: TyCtxt<'_>) -> U {
        U::from_tcx(self, tcx)
    }
}

impl<I, T, U> FromWithTcx<I> for Vec<U>
where
    I: IntoIterator<Item = T>,
    U: FromWithTcx<T>,
{
    fn from_tcx(f: I, tcx: TyCtxt<'_>) -> Vec<U> {
        f.into_iter().map(|x| x.into_tcx(tcx)).collect()
    }
}

pub(crate) fn from_deprecation(deprecation: rustc_attr::Deprecation) -> Deprecation {
    #[rustfmt::skip]
    let rustc_attr::Deprecation { since, note, is_since_rustc_version: _, suggestion: _ } = deprecation;
    Deprecation { since: since.map(|s| s.to_string()), note: note.map(|s| s.to_string()) }
}

impl FromWithTcx<clean::GenericArgs> for GenericArgs {
    fn from_tcx(args: clean::GenericArgs, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericArgs::*;
        match args {
            AngleBracketed { args, bindings } => GenericArgs::AngleBracketed {
                args: args.into_vec().into_tcx(tcx),
                bindings: bindings.into_tcx(tcx),
            },
            Parenthesized { inputs, output } => GenericArgs::Parenthesized {
                inputs: inputs.into_vec().into_tcx(tcx),
                output: output.map(|a| (*a).into_tcx(tcx)),
            },
        }
    }
}

impl FromWithTcx<clean::GenericArg> for GenericArg {
    fn from_tcx(arg: clean::GenericArg, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericArg::*;
        match arg {
            Lifetime(l) => GenericArg::Lifetime(convert_lifetime(l)),
            Type(t) => GenericArg::Type(t.into_tcx(tcx)),
            Const(box c) => GenericArg::Const(c.into_tcx(tcx)),
            Infer => GenericArg::Infer,
        }
    }
}

impl FromWithTcx<clean::Constant> for Constant {
    fn from_tcx(constant: clean::Constant, tcx: TyCtxt<'_>) -> Self {
        let expr = constant.expr(tcx);
        let value = constant.value(tcx);
        let is_literal = constant.is_literal(tcx);
        Constant { type_: constant.type_.into_tcx(tcx), expr, value, is_literal }
    }
}

impl FromWithTcx<clean::TypeBinding> for TypeBinding {
    fn from_tcx(binding: clean::TypeBinding, tcx: TyCtxt<'_>) -> Self {
        TypeBinding {
            name: binding.assoc.name.to_string(),
            args: binding.assoc.args.into_tcx(tcx),
            binding: binding.kind.into_tcx(tcx),
        }
    }
}

impl FromWithTcx<clean::TypeBindingKind> for TypeBindingKind {
    fn from_tcx(kind: clean::TypeBindingKind, tcx: TyCtxt<'_>) -> Self {
        use clean::TypeBindingKind::*;
        match kind {
            Equality { term } => TypeBindingKind::Equality(term.into_tcx(tcx)),
            Constraint { bounds } => TypeBindingKind::Constraint(bounds.into_tcx(tcx)),
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

fn from_clean_item(item: clean::Item, tcx: TyCtxt<'_>) -> ItemEnum {
    use clean::ItemKind::*;
    let name = item.name;
    let is_crate = item.is_crate();
    let header = item.fn_header(tcx);

    match *item.kind {
        ModuleItem(m) => {
            ItemEnum::Module(Module { is_crate, items: ids(m.items, tcx), is_stripped: false })
        }
        ImportItem(i) => ItemEnum::Import(i.into_tcx(tcx)),
        StructItem(s) => ItemEnum::Struct(s.into_tcx(tcx)),
        UnionItem(u) => ItemEnum::Union(u.into_tcx(tcx)),
        StructFieldItem(f) => ItemEnum::StructField(f.into_tcx(tcx)),
        EnumItem(e) => ItemEnum::Enum(e.into_tcx(tcx)),
        VariantItem(v) => ItemEnum::Variant(v.into_tcx(tcx)),
        FunctionItem(f) => ItemEnum::Function(from_function(f, header.unwrap(), tcx)),
        ForeignFunctionItem(f) => ItemEnum::Function(from_function(f, header.unwrap(), tcx)),
        TraitItem(t) => ItemEnum::Trait((*t).into_tcx(tcx)),
        TraitAliasItem(t) => ItemEnum::TraitAlias(t.into_tcx(tcx)),
        MethodItem(m, _) => ItemEnum::Method(from_function_method(m, true, header.unwrap(), tcx)),
        TyMethodItem(m) => ItemEnum::Method(from_function_method(m, false, header.unwrap(), tcx)),
        ImplItem(i) => ItemEnum::Impl((*i).into_tcx(tcx)),
        StaticItem(s) => ItemEnum::Static(s.into_tcx(tcx)),
        ForeignStaticItem(s) => ItemEnum::Static(s.into_tcx(tcx)),
        ForeignTypeItem => ItemEnum::ForeignType,
        TypedefItem(t) => ItemEnum::Typedef(t.into_tcx(tcx)),
        OpaqueTyItem(t) => ItemEnum::OpaqueTy(t.into_tcx(tcx)),
        ConstantItem(c) => ItemEnum::Constant(c.into_tcx(tcx)),
        MacroItem(m) => ItemEnum::Macro(m.source),
        ProcMacroItem(m) => ItemEnum::ProcMacro(m.into_tcx(tcx)),
        PrimitiveItem(p) => ItemEnum::PrimitiveType(p.as_sym().to_string()),
        TyAssocConstItem(ty) => ItemEnum::AssocConst { type_: ty.into_tcx(tcx), default: None },
        AssocConstItem(ty, default) => {
            ItemEnum::AssocConst { type_: ty.into_tcx(tcx), default: Some(default.expr(tcx)) }
        }
        TyAssocTypeItem(g, b) => ItemEnum::AssocType {
            generics: (*g).into_tcx(tcx),
            bounds: b.into_tcx(tcx),
            default: None,
        },
        AssocTypeItem(t, b) => ItemEnum::AssocType {
            generics: t.generics.into_tcx(tcx),
            bounds: b.into_tcx(tcx),
            default: Some(t.item_type.unwrap_or(t.type_).into_tcx(tcx)),
        },
        // `convert_item` early returns `None` for stripped items and keywords.
        KeywordItem => unreachable!(),
        StrippedItem(inner) => {
            match *inner {
                ModuleItem(m) => ItemEnum::Module(Module {
                    is_crate,
                    items: ids(m.items, tcx),
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

impl FromWithTcx<clean::Struct> for Struct {
    fn from_tcx(struct_: clean::Struct, tcx: TyCtxt<'_>) -> Self {
        let fields_stripped = struct_.has_stripped_entries();
        let clean::Struct { struct_type, generics, fields } = struct_;

        let kind = match struct_type {
            CtorKind::Fn => StructKind::Tuple(ids_keeping_stripped(fields, tcx)),
            CtorKind::Const => {
                assert!(fields.is_empty());
                StructKind::Unit
            }
            CtorKind::Fictive => StructKind::Plain { fields: ids(fields, tcx), fields_stripped },
        };

        Struct {
            kind,
            generics: generics.into_tcx(tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::Union> for Union {
    fn from_tcx(union_: clean::Union, tcx: TyCtxt<'_>) -> Self {
        let fields_stripped = union_.has_stripped_entries();
        let clean::Union { generics, fields } = union_;
        Union {
            generics: generics.into_tcx(tcx),
            fields_stripped,
            fields: ids(fields, tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
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

impl FromWithTcx<clean::Generics> for Generics {
    fn from_tcx(generics: clean::Generics, tcx: TyCtxt<'_>) -> Self {
        Generics {
            params: generics.params.into_tcx(tcx),
            where_predicates: generics.where_predicates.into_tcx(tcx),
        }
    }
}

impl FromWithTcx<clean::GenericParamDef> for GenericParamDef {
    fn from_tcx(generic_param: clean::GenericParamDef, tcx: TyCtxt<'_>) -> Self {
        GenericParamDef {
            name: generic_param.name.to_string(),
            kind: generic_param.kind.into_tcx(tcx),
        }
    }
}

impl FromWithTcx<clean::GenericParamDefKind> for GenericParamDefKind {
    fn from_tcx(kind: clean::GenericParamDefKind, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericParamDefKind::*;
        match kind {
            Lifetime { outlives } => GenericParamDefKind::Lifetime {
                outlives: outlives.into_iter().map(convert_lifetime).collect(),
            },
            Type { did: _, bounds, default, synthetic } => GenericParamDefKind::Type {
                bounds: bounds.into_tcx(tcx),
                default: default.map(|x| (*x).into_tcx(tcx)),
                synthetic,
            },
            Const { did: _, ty, default } => GenericParamDefKind::Const {
                type_: (*ty).into_tcx(tcx),
                default: default.map(|x| *x),
            },
        }
    }
}

impl FromWithTcx<clean::WherePredicate> for WherePredicate {
    fn from_tcx(predicate: clean::WherePredicate, tcx: TyCtxt<'_>) -> Self {
        use clean::WherePredicate::*;
        match predicate {
            BoundPredicate { ty, bounds, bound_params } => WherePredicate::BoundPredicate {
                type_: ty.into_tcx(tcx),
                bounds: bounds.into_tcx(tcx),
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
                bounds: bounds.into_tcx(tcx),
            },
            EqPredicate { lhs, rhs } => {
                WherePredicate::EqPredicate { lhs: lhs.into_tcx(tcx), rhs: rhs.into_tcx(tcx) }
            }
        }
    }
}

impl FromWithTcx<clean::GenericBound> for GenericBound {
    fn from_tcx(bound: clean::GenericBound, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericBound::*;
        match bound {
            TraitBound(clean::PolyTrait { trait_, generic_params }, modifier) => {
                GenericBound::TraitBound {
                    trait_: trait_.into_tcx(tcx),
                    generic_params: generic_params.into_tcx(tcx),
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

impl FromWithTcx<clean::Type> for Type {
    fn from_tcx(ty: clean::Type, tcx: TyCtxt<'_>) -> Self {
        use clean::Type::{
            Array, BareFunction, BorrowedRef, Generic, ImplTrait, Infer, Primitive, QPath,
            RawPointer, Slice, Tuple,
        };

        match ty {
            clean::Type::Path { path } => Type::ResolvedPath(path.into_tcx(tcx)),
            clean::Type::DynTrait(bounds, lt) => Type::DynTrait(DynTrait {
                lifetime: lt.map(convert_lifetime),
                traits: bounds.into_tcx(tcx),
            }),
            Generic(s) => Type::Generic(s.to_string()),
            Primitive(p) => Type::Primitive(p.as_sym().to_string()),
            BareFunction(f) => Type::FunctionPointer(Box::new((*f).into_tcx(tcx))),
            Tuple(t) => Type::Tuple(t.into_tcx(tcx)),
            Slice(t) => Type::Slice(Box::new((*t).into_tcx(tcx))),
            Array(t, s) => Type::Array { type_: Box::new((*t).into_tcx(tcx)), len: s },
            ImplTrait(g) => Type::ImplTrait(g.into_tcx(tcx)),
            Infer => Type::Infer,
            RawPointer(mutability, type_) => Type::RawPointer {
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_tcx(tcx)),
            },
            BorrowedRef { lifetime, mutability, type_ } => Type::BorrowedRef {
                lifetime: lifetime.map(convert_lifetime),
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_tcx(tcx)),
            },
            QPath(box clean::QPathData { assoc, self_type, trait_, .. }) => Type::QualifiedPath {
                name: assoc.name.to_string(),
                args: Box::new(assoc.args.into_tcx(tcx)),
                self_type: Box::new(self_type.into_tcx(tcx)),
                trait_: trait_.into_tcx(tcx),
            },
        }
    }
}

impl FromWithTcx<clean::Path> for Path {
    fn from_tcx(path: clean::Path, tcx: TyCtxt<'_>) -> Path {
        Path {
            name: path.whole_name(),
            id: from_item_id(path.def_id().into(), tcx),
            args: path.segments.last().map(|args| Box::new(args.clone().args.into_tcx(tcx))),
        }
    }
}

impl FromWithTcx<clean::Term> for Term {
    fn from_tcx(term: clean::Term, tcx: TyCtxt<'_>) -> Term {
        match term {
            clean::Term::Type(ty) => Term::Type(FromWithTcx::from_tcx(ty, tcx)),
            clean::Term::Constant(c) => Term::Constant(FromWithTcx::from_tcx(c, tcx)),
        }
    }
}

impl FromWithTcx<clean::BareFunctionDecl> for FunctionPointer {
    fn from_tcx(bare_decl: clean::BareFunctionDecl, tcx: TyCtxt<'_>) -> Self {
        let clean::BareFunctionDecl { unsafety, generic_params, decl, abi } = bare_decl;
        FunctionPointer {
            header: Header {
                unsafe_: matches!(unsafety, rustc_hir::Unsafety::Unsafe),
                const_: false,
                async_: false,
                abi: convert_abi(abi),
            },
            generic_params: generic_params.into_tcx(tcx),
            decl: decl.into_tcx(tcx),
        }
    }
}

impl FromWithTcx<clean::FnDecl> for FnDecl {
    fn from_tcx(decl: clean::FnDecl, tcx: TyCtxt<'_>) -> Self {
        let clean::FnDecl { inputs, output, c_variadic } = decl;
        FnDecl {
            inputs: inputs
                .values
                .into_iter()
                .map(|arg| (arg.name.to_string(), arg.type_.into_tcx(tcx)))
                .collect(),
            output: match output {
                clean::FnRetTy::Return(t) => Some(t.into_tcx(tcx)),
                clean::FnRetTy::DefaultReturn => None,
            },
            c_variadic,
        }
    }
}

impl FromWithTcx<clean::Trait> for Trait {
    fn from_tcx(trait_: clean::Trait, tcx: TyCtxt<'_>) -> Self {
        let is_auto = trait_.is_auto(tcx);
        let is_unsafe = trait_.unsafety(tcx) == rustc_hir::Unsafety::Unsafe;
        let clean::Trait { items, generics, bounds, .. } = trait_;
        Trait {
            is_auto,
            is_unsafe,
            items: ids(items, tcx),
            generics: generics.into_tcx(tcx),
            bounds: bounds.into_tcx(tcx),
            implementations: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::PolyTrait> for PolyTrait {
    fn from_tcx(
        clean::PolyTrait { trait_, generic_params }: clean::PolyTrait,
        tcx: TyCtxt<'_>,
    ) -> Self {
        PolyTrait { trait_: trait_.into_tcx(tcx), generic_params: generic_params.into_tcx(tcx) }
    }
}

impl FromWithTcx<clean::Impl> for Impl {
    fn from_tcx(impl_: clean::Impl, tcx: TyCtxt<'_>) -> Self {
        let provided_trait_methods = impl_.provided_trait_methods(tcx);
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
            generics: generics.into_tcx(tcx),
            provided_trait_methods: provided_trait_methods
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            trait_: trait_.map(|path| path.into_tcx(tcx)),
            for_: for_.into_tcx(tcx),
            items: ids(items, tcx),
            negative: negative_polarity,
            synthetic,
            blanket_impl: blanket_impl.map(|x| x.into_tcx(tcx)),
        }
    }
}

pub(crate) fn from_function(
    function: Box<clean::Function>,
    header: rustc_hir::FnHeader,
    tcx: TyCtxt<'_>,
) -> Function {
    let clean::Function { decl, generics } = *function;
    Function {
        decl: decl.into_tcx(tcx),
        generics: generics.into_tcx(tcx),
        header: from_fn_header(&header),
    }
}

pub(crate) fn from_function_method(
    function: Box<clean::Function>,
    has_body: bool,
    header: rustc_hir::FnHeader,
    tcx: TyCtxt<'_>,
) -> Method {
    let clean::Function { decl, generics } = *function;
    Method {
        decl: decl.into_tcx(tcx),
        generics: generics.into_tcx(tcx),
        header: from_fn_header(&header),
        has_body,
    }
}

impl FromWithTcx<clean::Enum> for Enum {
    fn from_tcx(enum_: clean::Enum, tcx: TyCtxt<'_>) -> Self {
        let variants_stripped = enum_.has_stripped_entries();
        let clean::Enum { variants, generics } = enum_;
        Enum {
            generics: generics.into_tcx(tcx),
            variants_stripped,
            variants: ids(variants, tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::Variant> for Variant {
    fn from_tcx(variant: clean::Variant, tcx: TyCtxt<'_>) -> Self {
        use clean::Variant::*;
        match variant {
            CLike(disr) => Variant::Plain(disr.map(|disr| disr.into_tcx(tcx))),
            Tuple(fields) => Variant::Tuple(ids_keeping_stripped(fields, tcx)),
            Struct(s) => Variant::Struct {
                fields_stripped: s.has_stripped_entries(),
                fields: ids(s.fields, tcx),
            },
        }
    }
}

impl FromWithTcx<clean::Discriminant> for Discriminant {
    fn from_tcx(disr: clean::Discriminant, tcx: TyCtxt<'_>) -> Self {
        Discriminant {
            // expr is only none if going throught the inlineing path, which gets
            // `rustc_middle` types, not `rustc_hir`, but because JSON never inlines
            // the expr is always some.
            expr: disr.expr(tcx).unwrap(),
            value: disr.value(tcx),
        }
    }
}

impl FromWithTcx<clean::Import> for Import {
    fn from_tcx(import: clean::Import, tcx: TyCtxt<'_>) -> Self {
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
            id: import.source.did.map(ItemId::from).map(|i| from_item_id(i, tcx)),
            glob,
        }
    }
}

impl FromWithTcx<clean::ProcMacro> for ProcMacro {
    fn from_tcx(mac: clean::ProcMacro, _tcx: TyCtxt<'_>) -> Self {
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

impl FromWithTcx<Box<clean::Typedef>> for Typedef {
    fn from_tcx(typedef: Box<clean::Typedef>, tcx: TyCtxt<'_>) -> Self {
        let clean::Typedef { type_, generics, item_type: _ } = *typedef;
        Typedef { type_: type_.into_tcx(tcx), generics: generics.into_tcx(tcx) }
    }
}

impl FromWithTcx<clean::OpaqueTy> for OpaqueTy {
    fn from_tcx(opaque: clean::OpaqueTy, tcx: TyCtxt<'_>) -> Self {
        OpaqueTy { bounds: opaque.bounds.into_tcx(tcx), generics: opaque.generics.into_tcx(tcx) }
    }
}

impl FromWithTcx<clean::Static> for Static {
    fn from_tcx(stat: clean::Static, tcx: TyCtxt<'_>) -> Self {
        Static {
            type_: stat.type_.into_tcx(tcx),
            mutable: stat.mutability == ast::Mutability::Mut,
            expr: stat.expr.map(|e| print_const_expr(tcx, e)).unwrap_or_default(),
        }
    }
}

impl FromWithTcx<clean::TraitAlias> for TraitAlias {
    fn from_tcx(alias: clean::TraitAlias, tcx: TyCtxt<'_>) -> Self {
        TraitAlias { generics: alias.generics.into_tcx(tcx), params: alias.bounds.into_tcx(tcx) }
    }
}

impl FromWithTcx<ItemType> for ItemKind {
    fn from_tcx(kind: ItemType, _tcx: TyCtxt<'_>) -> Self {
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

fn ids_keeping_stripped(
    items: impl IntoIterator<Item = clean::Item>,
    tcx: TyCtxt<'_>,
) -> Vec<Option<Id>> {
    items
        .into_iter()
        .map(|i| {
            if !i.is_stripped() && !i.is_keyword() {
                Some(from_item_id_with_name(i.item_id, tcx, i.name))
            } else {
                None
            }
        })
        .collect()
}
