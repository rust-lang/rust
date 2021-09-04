//! These from impls are used to create the JSON types which get serialized. They're very close to
//! the `clean` types but with some fields removed or stringified to simplify the output and not
//! expose unstable compiler internals.

#![allow(rustc::default_hash_types)]

use std::convert::From;
use std::fmt;

use rustc_ast::ast;
use rustc_hir::{def::CtorKind, def_id::DefId};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::CRATE_DEF_INDEX;
use rustc_span::Pos;

use rustdoc_json_types::*;

use crate::clean::utils::print_const_expr;
use crate::clean::{self, ItemId};
use crate::formats::item_type::ItemType;
use crate::json::JsonRenderer;
use std::collections::HashSet;

impl JsonRenderer<'_> {
    pub(super) fn convert_item(&self, item: clean::Item) -> Option<Item> {
        let deprecation = item.deprecation(self.tcx);
        let links = self
            .cache
            .intra_doc_links
            .get(&item.def_id)
            .into_iter()
            .flatten()
            .filter_map(|clean::ItemLink { link, did, .. }| {
                did.map(|did| (link.clone(), from_item_id(did.into())))
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
        let clean::Item { name, attrs: _, kind: _, visibility, def_id, cfg: _ } = item;
        let inner = match *item.kind {
            clean::StrippedItem(_) => return None,
            _ => from_clean_item(item, self.tcx),
        };
        Some(Item {
            id: from_item_id(def_id),
            crate_id: def_id.krate().as_u32(),
            name: name.map(|sym| sym.to_string()),
            span: self.convert_span(span),
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
            Restricted(did) if did.index == CRATE_DEF_INDEX => Visibility::Crate,
            Restricted(did) => Visibility::Restricted {
                parent: from_item_id(did.into()),
                path: self.tcx.def_path(did).to_string_no_crate_verbose(),
            },
        }
    }
}

crate trait FromWithTcx<T> {
    fn from_tcx(f: T, tcx: TyCtxt<'_>) -> Self;
}

crate trait IntoWithTcx<T> {
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

crate fn from_deprecation(deprecation: rustc_attr::Deprecation) -> Deprecation {
    #[rustfmt::skip]
    let rustc_attr::Deprecation { since, note, is_since_rustc_version: _, suggestion: _ } = deprecation;
    Deprecation { since: since.map(|s| s.to_string()), note: note.map(|s| s.to_string()) }
}

impl FromWithTcx<clean::GenericArgs> for GenericArgs {
    fn from_tcx(args: clean::GenericArgs, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericArgs::*;
        match args {
            AngleBracketed { args, bindings } => GenericArgs::AngleBracketed {
                args: args.into_iter().map(|a| a.into_tcx(tcx)).collect(),
                bindings: bindings.into_iter().map(|a| a.into_tcx(tcx)).collect(),
            },
            Parenthesized { inputs, output } => GenericArgs::Parenthesized {
                inputs: inputs.into_iter().map(|a| a.into_tcx(tcx)).collect(),
                output: output.map(|a| (*a).into_tcx(tcx)),
            },
        }
    }
}

impl FromWithTcx<clean::GenericArg> for GenericArg {
    fn from_tcx(arg: clean::GenericArg, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericArg::*;
        match arg {
            Lifetime(l) => GenericArg::Lifetime(l.0.to_string()),
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
        TypeBinding { name: binding.name.to_string(), binding: binding.kind.into_tcx(tcx) }
    }
}

impl FromWithTcx<clean::TypeBindingKind> for TypeBindingKind {
    fn from_tcx(kind: clean::TypeBindingKind, tcx: TyCtxt<'_>) -> Self {
        use clean::TypeBindingKind::*;
        match kind {
            Equality { ty } => TypeBindingKind::Equality(ty.into_tcx(tcx)),
            Constraint { bounds } => {
                TypeBindingKind::Constraint(bounds.into_iter().map(|a| a.into_tcx(tcx)).collect())
            }
        }
    }
}

crate fn from_item_id(did: ItemId) -> Id {
    struct DisplayDefId(DefId);

    impl fmt::Display for DisplayDefId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}:{}", self.0.krate.as_u32(), u32::from(self.0.index))
        }
    }

    match did {
        ItemId::DefId(did) => Id(format!("{}", DisplayDefId(did))),
        ItemId::Blanket { for_, impl_id } => {
            Id(format!("b:{}-{}", DisplayDefId(impl_id), DisplayDefId(for_)))
        }
        ItemId::Auto { for_, trait_ } => {
            Id(format!("a:{}-{}", DisplayDefId(trait_), DisplayDefId(for_)))
        }
        ItemId::Primitive(ty, krate) => Id(format!("p:{}:{}", krate.as_u32(), ty.as_sym())),
    }
}

fn from_clean_item(item: clean::Item, tcx: TyCtxt<'_>) -> ItemEnum {
    use clean::ItemKind::*;
    let name = item.name;
    let is_crate = item.is_crate();
    match *item.kind {
        ModuleItem(m) => ItemEnum::Module(Module { is_crate, items: ids(m.items) }),
        ImportItem(i) => ItemEnum::Import(i.into_tcx(tcx)),
        StructItem(s) => ItemEnum::Struct(s.into_tcx(tcx)),
        UnionItem(u) => ItemEnum::Union(u.into_tcx(tcx)),
        StructFieldItem(f) => ItemEnum::StructField(f.into_tcx(tcx)),
        EnumItem(e) => ItemEnum::Enum(e.into_tcx(tcx)),
        VariantItem(v) => ItemEnum::Variant(v.into_tcx(tcx)),
        FunctionItem(f) => ItemEnum::Function(f.into_tcx(tcx)),
        ForeignFunctionItem(f) => ItemEnum::Function(f.into_tcx(tcx)),
        TraitItem(t) => ItemEnum::Trait(t.into_tcx(tcx)),
        TraitAliasItem(t) => ItemEnum::TraitAlias(t.into_tcx(tcx)),
        MethodItem(m, _) => ItemEnum::Method(from_function_method(m, true, tcx)),
        TyMethodItem(m) => ItemEnum::Method(from_function_method(m, false, tcx)),
        ImplItem(i) => ItemEnum::Impl(i.into_tcx(tcx)),
        StaticItem(s) => ItemEnum::Static(s.into_tcx(tcx)),
        ForeignStaticItem(s) => ItemEnum::Static(s.into_tcx(tcx)),
        ForeignTypeItem => ItemEnum::ForeignType,
        TypedefItem(t, _) => ItemEnum::Typedef(t.into_tcx(tcx)),
        OpaqueTyItem(t) => ItemEnum::OpaqueTy(t.into_tcx(tcx)),
        ConstantItem(c) => ItemEnum::Constant(c.into_tcx(tcx)),
        MacroItem(m) => ItemEnum::Macro(m.source),
        ProcMacroItem(m) => ItemEnum::ProcMacro(m.into_tcx(tcx)),
        AssocConstItem(t, s) => ItemEnum::AssocConst { type_: t.into_tcx(tcx), default: s },
        AssocTypeItem(g, t) => ItemEnum::AssocType {
            bounds: g.into_iter().map(|x| x.into_tcx(tcx)).collect(),
            default: t.map(|x| x.into_tcx(tcx)),
        },
        // `convert_item` early returns `None` for striped items
        StrippedItem(_) => unreachable!(),
        PrimitiveItem(_) | KeywordItem(_) => {
            panic!("{:?} is not supported for JSON output", item)
        }
        ExternCrateItem { ref src } => ItemEnum::ExternCrate {
            name: name.as_ref().unwrap().to_string(),
            rename: src.map(|x| x.to_string()),
        },
    }
}

impl FromWithTcx<clean::Struct> for Struct {
    fn from_tcx(struct_: clean::Struct, tcx: TyCtxt<'_>) -> Self {
        let clean::Struct { struct_type, generics, fields, fields_stripped } = struct_;
        Struct {
            struct_type: from_ctor_kind(struct_type),
            generics: generics.into_tcx(tcx),
            fields_stripped,
            fields: ids(fields),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::Union> for Union {
    fn from_tcx(struct_: clean::Union, tcx: TyCtxt<'_>) -> Self {
        let clean::Union { generics, fields, fields_stripped } = struct_;
        Union {
            generics: generics.into_tcx(tcx),
            fields_stripped,
            fields: ids(fields),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

crate fn from_ctor_kind(struct_type: CtorKind) -> StructType {
    match struct_type {
        CtorKind::Fictive => StructType::Plain,
        CtorKind::Fn => StructType::Tuple,
        CtorKind::Const => StructType::Unit,
    }
}

crate fn from_fn_header(header: &rustc_hir::FnHeader) -> HashSet<Qualifiers> {
    let mut v = HashSet::new();

    if let rustc_hir::Unsafety::Unsafe = header.unsafety {
        v.insert(Qualifiers::Unsafe);
    }

    if let rustc_hir::IsAsync::Async = header.asyncness {
        v.insert(Qualifiers::Async);
    }

    if let rustc_hir::Constness::Const = header.constness {
        v.insert(Qualifiers::Const);
    }

    v
}

impl FromWithTcx<clean::Function> for Function {
    fn from_tcx(function: clean::Function, tcx: TyCtxt<'_>) -> Self {
        let clean::Function { decl, generics, header } = function;
        Function {
            decl: decl.into_tcx(tcx),
            generics: generics.into_tcx(tcx),
            header: from_fn_header(&header),
            abi: header.abi.to_string(),
        }
    }
}

impl FromWithTcx<clean::Generics> for Generics {
    fn from_tcx(generics: clean::Generics, tcx: TyCtxt<'_>) -> Self {
        Generics {
            params: generics.params.into_iter().map(|x| x.into_tcx(tcx)).collect(),
            where_predicates: generics
                .where_predicates
                .into_iter()
                .map(|x| x.into_tcx(tcx))
                .collect(),
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
            Lifetime => GenericParamDefKind::Lifetime,
            Type { did: _, bounds, default, synthetic: _ } => GenericParamDefKind::Type {
                bounds: bounds.into_iter().map(|x| x.into_tcx(tcx)).collect(),
                default: default.map(|x| x.into_tcx(tcx)),
            },
            Const { did: _, ty, default } => {
                GenericParamDefKind::Const { ty: ty.into_tcx(tcx), default }
            }
        }
    }
}

impl FromWithTcx<clean::WherePredicate> for WherePredicate {
    fn from_tcx(predicate: clean::WherePredicate, tcx: TyCtxt<'_>) -> Self {
        use clean::WherePredicate::*;
        match predicate {
            BoundPredicate { ty, bounds, .. } => WherePredicate::BoundPredicate {
                ty: ty.into_tcx(tcx),
                bounds: bounds.into_iter().map(|x| x.into_tcx(tcx)).collect(),
                // FIXME: add `bound_params` to rustdoc-json-params?
            },
            RegionPredicate { lifetime, bounds } => WherePredicate::RegionPredicate {
                lifetime: lifetime.0.to_string(),
                bounds: bounds.into_iter().map(|x| x.into_tcx(tcx)).collect(),
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
                    generic_params: generic_params.into_iter().map(|x| x.into_tcx(tcx)).collect(),
                    modifier: from_trait_bound_modifier(modifier),
                }
            }
            Outlives(lifetime) => GenericBound::Outlives(lifetime.0.to_string()),
        }
    }
}

crate fn from_trait_bound_modifier(modifier: rustc_hir::TraitBoundModifier) -> TraitBoundModifier {
    use rustc_hir::TraitBoundModifier::*;
    match modifier {
        None => TraitBoundModifier::None,
        Maybe => TraitBoundModifier::Maybe,
        MaybeConst => TraitBoundModifier::MaybeConst,
    }
}

impl FromWithTcx<clean::Type> for Type {
    fn from_tcx(ty: clean::Type, tcx: TyCtxt<'_>) -> Self {
        use clean::Type::*;
        match ty {
            ResolvedPath { path, did, is_generic: _ } => Type::ResolvedPath {
                name: path.whole_name(),
                id: from_item_id(did.into()),
                args: path.segments.last().map(|args| Box::new(args.clone().args.into_tcx(tcx))),
                param_names: Vec::new(),
            },
            DynTrait(mut bounds, lt) => {
                let (path, id) = match bounds.remove(0).trait_ {
                    ResolvedPath { path, did, .. } => (path, did),
                    _ => unreachable!(),
                };

                Type::ResolvedPath {
                    name: path.whole_name(),
                    id: from_item_id(id.into()),
                    args: path
                        .segments
                        .last()
                        .map(|args| Box::new(args.clone().args.into_tcx(tcx))),
                    param_names: bounds
                        .into_iter()
                        .map(|t| {
                            clean::GenericBound::TraitBound(t, rustc_hir::TraitBoundModifier::None)
                        })
                        .chain(lt.into_iter().map(|lt| clean::GenericBound::Outlives(lt)))
                        .map(|bound| bound.into_tcx(tcx))
                        .collect(),
                }
            }
            Generic(s) => Type::Generic(s.to_string()),
            Primitive(p) => Type::Primitive(p.as_sym().to_string()),
            BareFunction(f) => Type::FunctionPointer(Box::new((*f).into_tcx(tcx))),
            Tuple(t) => Type::Tuple(t.into_iter().map(|x| x.into_tcx(tcx)).collect()),
            Slice(t) => Type::Slice(Box::new((*t).into_tcx(tcx))),
            Array(t, s) => Type::Array { type_: Box::new((*t).into_tcx(tcx)), len: s },
            ImplTrait(g) => Type::ImplTrait(g.into_iter().map(|x| x.into_tcx(tcx)).collect()),
            Never => Type::Never,
            Infer => Type::Infer,
            RawPointer(mutability, type_) => Type::RawPointer {
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_tcx(tcx)),
            },
            BorrowedRef { lifetime, mutability, type_ } => Type::BorrowedRef {
                lifetime: lifetime.map(|l| l.0.to_string()),
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_tcx(tcx)),
            },
            QPath { name, self_type, trait_, .. } => Type::QualifiedPath {
                name: name.to_string(),
                self_type: Box::new((*self_type).into_tcx(tcx)),
                trait_: Box::new((*trait_).into_tcx(tcx)),
            },
        }
    }
}

impl FromWithTcx<clean::BareFunctionDecl> for FunctionPointer {
    fn from_tcx(bare_decl: clean::BareFunctionDecl, tcx: TyCtxt<'_>) -> Self {
        let clean::BareFunctionDecl { unsafety, generic_params, decl, abi } = bare_decl;
        FunctionPointer {
            header: if let rustc_hir::Unsafety::Unsafe = unsafety {
                let mut hs = HashSet::new();
                hs.insert(Qualifiers::Unsafe);
                hs
            } else {
                HashSet::new()
            },
            generic_params: generic_params.into_iter().map(|x| x.into_tcx(tcx)).collect(),
            decl: decl.into_tcx(tcx),
            abi: abi.to_string(),
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
        let clean::Trait { unsafety, items, generics, bounds, is_auto } = trait_;
        Trait {
            is_auto,
            is_unsafe: unsafety == rustc_hir::Unsafety::Unsafe,
            items: ids(items),
            generics: generics.into_tcx(tcx),
            bounds: bounds.into_iter().map(|x| x.into_tcx(tcx)).collect(),
            implementors: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::Impl> for Impl {
    fn from_tcx(impl_: clean::Impl, tcx: TyCtxt<'_>) -> Self {
        let provided_trait_methods = impl_.provided_trait_methods(tcx);
        let clean::Impl {
            unsafety,
            generics,
            trait_,
            for_,
            items,
            negative_polarity,
            synthetic,
            blanket_impl,
            span: _span,
        } = impl_;
        Impl {
            is_unsafe: unsafety == rustc_hir::Unsafety::Unsafe,
            generics: generics.into_tcx(tcx),
            provided_trait_methods: provided_trait_methods
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            trait_: trait_.map(|x| x.into_tcx(tcx)),
            for_: for_.into_tcx(tcx),
            items: ids(items),
            negative: negative_polarity,
            synthetic,
            blanket_impl: blanket_impl.map(|x| (*x).into_tcx(tcx)),
        }
    }
}

crate fn from_function_method(
    function: clean::Function,
    has_body: bool,
    tcx: TyCtxt<'_>,
) -> Method {
    let clean::Function { header, decl, generics } = function;
    Method {
        decl: decl.into_tcx(tcx),
        generics: generics.into_tcx(tcx),
        header: from_fn_header(&header),
        abi: header.abi.to_string(),
        has_body,
    }
}

impl FromWithTcx<clean::Enum> for Enum {
    fn from_tcx(enum_: clean::Enum, tcx: TyCtxt<'_>) -> Self {
        let clean::Enum { variants, generics, variants_stripped } = enum_;
        Enum {
            generics: generics.into_tcx(tcx),
            variants_stripped,
            variants: ids(variants),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::VariantStruct> for Struct {
    fn from_tcx(struct_: clean::VariantStruct, _tcx: TyCtxt<'_>) -> Self {
        let clean::VariantStruct { struct_type, fields, fields_stripped } = struct_;
        Struct {
            struct_type: from_ctor_kind(struct_type),
            generics: Default::default(),
            fields_stripped,
            fields: ids(fields),
            impls: Vec::new(),
        }
    }
}

impl FromWithTcx<clean::Variant> for Variant {
    fn from_tcx(variant: clean::Variant, tcx: TyCtxt<'_>) -> Self {
        use clean::Variant::*;
        match variant {
            CLike => Variant::Plain,
            Tuple(fields) => Variant::Tuple(
                fields
                    .into_iter()
                    .map(|f| {
                        if let clean::StructFieldItem(ty) = *f.kind {
                            ty.into_tcx(tcx)
                        } else {
                            unreachable!()
                        }
                    })
                    .collect(),
            ),
            Struct(s) => Variant::Struct(ids(s.fields)),
        }
    }
}

impl FromWithTcx<clean::Import> for Import {
    fn from_tcx(import: clean::Import, _tcx: TyCtxt<'_>) -> Self {
        use clean::ImportKind::*;
        match import.kind {
            Simple(s) => Import {
                source: import.source.path.whole_name(),
                name: s.to_string(),
                id: import.source.did.map(ItemId::from).map(from_item_id),
                glob: false,
            },
            Glob => Import {
                source: import.source.path.whole_name(),
                name: import.source.path.last_name().to_string(),
                id: import.source.did.map(ItemId::from).map(from_item_id),
                glob: true,
            },
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

crate fn from_macro_kind(kind: rustc_span::hygiene::MacroKind) -> MacroKind {
    use rustc_span::hygiene::MacroKind::*;
    match kind {
        Bang => MacroKind::Bang,
        Attr => MacroKind::Attr,
        Derive => MacroKind::Derive,
    }
}

impl FromWithTcx<clean::Typedef> for Typedef {
    fn from_tcx(typedef: clean::Typedef, tcx: TyCtxt<'_>) -> Self {
        let clean::Typedef { type_, generics, item_type: _ } = typedef;
        Typedef { type_: type_.into_tcx(tcx), generics: generics.into_tcx(tcx) }
    }
}

impl FromWithTcx<clean::OpaqueTy> for OpaqueTy {
    fn from_tcx(opaque: clean::OpaqueTy, tcx: TyCtxt<'_>) -> Self {
        OpaqueTy {
            bounds: opaque.bounds.into_iter().map(|x| x.into_tcx(tcx)).collect(),
            generics: opaque.generics.into_tcx(tcx),
        }
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
        TraitAlias {
            generics: alias.generics.into_tcx(tcx),
            params: alias.bounds.into_iter().map(|x| x.into_tcx(tcx)).collect(),
        }
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

fn ids(items: impl IntoIterator<Item = clean::Item>) -> Vec<Id> {
    items.into_iter().filter(|x| !x.is_stripped()).map(|i| from_item_id(i.def_id)).collect()
}
