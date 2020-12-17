//! These from impls are used to create the JSON types which get serialized. They're very close to
//! the `clean` types but with some fields removed or stringified to simplify the output and not
//! expose unstable compiler internals.

use std::convert::From;

use rustc_ast::ast;
use rustc_span::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_span::Pos;

use crate::clean;
use crate::doctree;
use crate::formats::item_type::ItemType;
use crate::json::types::*;
use crate::json::JsonRenderer;

impl JsonRenderer {
    pub(super) fn convert_item(&self, item: clean::Item) -> Option<Item> {
        let item_type = ItemType::from(&item);
        let clean::Item {
            source,
            name,
            attrs,
            kind,
            visibility,
            def_id,
            stability: _,
            const_stability: _,
            deprecation,
        } = item;
        match kind {
            clean::StrippedItem(_) => None,
            _ => Some(Item {
                id: def_id.into(),
                crate_id: def_id.krate.as_u32(),
                name: name.map(|sym| sym.to_string()),
                source: self.convert_span(source),
                visibility: visibility.into(),
                docs: attrs.collapsed_doc_value().unwrap_or_default(),
                links: attrs
                    .links
                    .into_iter()
                    .filter_map(|clean::ItemLink { link, did, .. }| {
                        did.map(|did| (link, did.into()))
                    })
                    .collect(),
                attrs: attrs
                    .other_attrs
                    .iter()
                    .map(rustc_ast_pretty::pprust::attribute_to_string)
                    .collect(),
                deprecation: deprecation.map(Into::into),
                kind: item_type.into(),
                inner: kind.into(),
            }),
        }
    }

    fn convert_span(&self, span: clean::Span) -> Option<Span> {
        match span.filename(&self.sess) {
            rustc_span::FileName::Real(name) => {
                let hi = span.hi(&self.sess);
                let lo = span.lo(&self.sess);
                Some(Span {
                    filename: match name {
                        rustc_span::RealFileName::Named(path) => path,
                        rustc_span::RealFileName::Devirtualized { local_path, virtual_name: _ } => {
                            local_path
                        }
                    },
                    begin: (lo.line, lo.col.to_usize()),
                    end: (hi.line, hi.col.to_usize()),
                })
            }
            _ => None,
        }
    }
}

impl From<rustc_attr::Deprecation> for Deprecation {
    fn from(deprecation: rustc_attr::Deprecation) -> Self {
        #[rustfmt::skip]
        let rustc_attr::Deprecation { since, note, is_since_rustc_version: _, suggestion: _ } = deprecation;
        Deprecation { since: since.map(|s| s.to_string()), note: note.map(|s| s.to_string()) }
    }
}

impl From<clean::Visibility> for Visibility {
    fn from(v: clean::Visibility) -> Self {
        use clean::Visibility::*;
        match v {
            Public => Visibility::Public,
            Inherited => Visibility::Default,
            Restricted(did, _) if did.index == CRATE_DEF_INDEX => Visibility::Crate,
            Restricted(did, path) => Visibility::Restricted {
                parent: did.into(),
                path: path.to_string_no_crate_verbose(),
            },
        }
    }
}

impl From<clean::GenericArgs> for GenericArgs {
    fn from(args: clean::GenericArgs) -> Self {
        use clean::GenericArgs::*;
        match args {
            AngleBracketed { args, bindings } => GenericArgs::AngleBracketed {
                args: args.into_iter().map(Into::into).collect(),
                bindings: bindings.into_iter().map(Into::into).collect(),
            },
            Parenthesized { inputs, output } => GenericArgs::Parenthesized {
                inputs: inputs.into_iter().map(Into::into).collect(),
                output: output.map(Into::into),
            },
        }
    }
}

impl From<clean::GenericArg> for GenericArg {
    fn from(arg: clean::GenericArg) -> Self {
        use clean::GenericArg::*;
        match arg {
            Lifetime(l) => GenericArg::Lifetime(l.0),
            Type(t) => GenericArg::Type(t.into()),
            Const(c) => GenericArg::Const(c.into()),
        }
    }
}

impl From<clean::Constant> for Constant {
    fn from(constant: clean::Constant) -> Self {
        let clean::Constant { type_, expr, value, is_literal } = constant;
        Constant { type_: type_.into(), expr, value, is_literal }
    }
}

impl From<clean::TypeBinding> for TypeBinding {
    fn from(binding: clean::TypeBinding) -> Self {
        TypeBinding { name: binding.name.to_string(), binding: binding.kind.into() }
    }
}

impl From<clean::TypeBindingKind> for TypeBindingKind {
    fn from(kind: clean::TypeBindingKind) -> Self {
        use clean::TypeBindingKind::*;
        match kind {
            Equality { ty } => TypeBindingKind::Equality(ty.into()),
            Constraint { bounds } => {
                TypeBindingKind::Constraint(bounds.into_iter().map(Into::into).collect())
            }
        }
    }
}

impl From<DefId> for Id {
    fn from(did: DefId) -> Self {
        Id(format!("{}:{}", did.krate.as_u32(), u32::from(did.index)))
    }
}

impl From<clean::ItemKind> for ItemEnum {
    fn from(item: clean::ItemKind) -> Self {
        use clean::ItemKind::*;
        match item {
            ModuleItem(m) => ItemEnum::ModuleItem(m.into()),
            ExternCrateItem(c, a) => ItemEnum::ExternCrateItem { name: c, rename: a },
            ImportItem(i) => ItemEnum::ImportItem(i.into()),
            StructItem(s) => ItemEnum::StructItem(s.into()),
            UnionItem(u) => ItemEnum::StructItem(u.into()),
            StructFieldItem(f) => ItemEnum::StructFieldItem(f.into()),
            EnumItem(e) => ItemEnum::EnumItem(e.into()),
            VariantItem(v) => ItemEnum::VariantItem(v.into()),
            FunctionItem(f) => ItemEnum::FunctionItem(f.into()),
            ForeignFunctionItem(f) => ItemEnum::FunctionItem(f.into()),
            TraitItem(t) => ItemEnum::TraitItem(t.into()),
            TraitAliasItem(t) => ItemEnum::TraitAliasItem(t.into()),
            MethodItem(m, _) => ItemEnum::MethodItem(m.into()),
            TyMethodItem(m) => ItemEnum::MethodItem(m.into()),
            ImplItem(i) => ItemEnum::ImplItem(i.into()),
            StaticItem(s) => ItemEnum::StaticItem(s.into()),
            ForeignStaticItem(s) => ItemEnum::StaticItem(s.into()),
            ForeignTypeItem => ItemEnum::ForeignTypeItem,
            TypedefItem(t, _) => ItemEnum::TypedefItem(t.into()),
            OpaqueTyItem(t) => ItemEnum::OpaqueTyItem(t.into()),
            ConstantItem(c) => ItemEnum::ConstantItem(c.into()),
            MacroItem(m) => ItemEnum::MacroItem(m.source),
            ProcMacroItem(m) => ItemEnum::ProcMacroItem(m.into()),
            AssocConstItem(t, s) => ItemEnum::AssocConstItem { type_: t.into(), default: s },
            AssocTypeItem(g, t) => ItemEnum::AssocTypeItem {
                bounds: g.into_iter().map(Into::into).collect(),
                default: t.map(Into::into),
            },
            StrippedItem(inner) => (*inner).into(),
            PrimitiveItem(_) | KeywordItem(_) => {
                panic!("{:?} is not supported for JSON output", item)
            }
        }
    }
}

impl From<clean::Module> for Module {
    fn from(module: clean::Module) -> Self {
        Module { is_crate: module.is_crate, items: ids(module.items) }
    }
}

impl From<clean::Struct> for Struct {
    fn from(struct_: clean::Struct) -> Self {
        let clean::Struct { struct_type, generics, fields, fields_stripped } = struct_;
        Struct {
            struct_type: struct_type.into(),
            generics: generics.into(),
            fields_stripped,
            fields: ids(fields),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl From<clean::Union> for Struct {
    fn from(struct_: clean::Union) -> Self {
        let clean::Union { struct_type, generics, fields, fields_stripped } = struct_;
        Struct {
            struct_type: struct_type.into(),
            generics: generics.into(),
            fields_stripped,
            fields: ids(fields),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl From<doctree::StructType> for StructType {
    fn from(struct_type: doctree::StructType) -> Self {
        use doctree::StructType::*;
        match struct_type {
            Plain => StructType::Plain,
            Tuple => StructType::Tuple,
            Unit => StructType::Unit,
        }
    }
}

fn stringify_header(header: &rustc_hir::FnHeader) -> String {
    let mut s = String::from(header.unsafety.prefix_str());
    if header.asyncness == rustc_hir::IsAsync::Async {
        s.push_str("async ")
    }
    if header.constness == rustc_hir::Constness::Const {
        s.push_str("const ")
    }
    s
}

impl From<clean::Function> for Function {
    fn from(function: clean::Function) -> Self {
        let clean::Function { decl, generics, header, all_types: _, ret_types: _ } = function;
        Function {
            decl: decl.into(),
            generics: generics.into(),
            header: stringify_header(&header),
            abi: header.abi.to_string(),
        }
    }
}

impl From<clean::Generics> for Generics {
    fn from(generics: clean::Generics) -> Self {
        Generics {
            params: generics.params.into_iter().map(Into::into).collect(),
            where_predicates: generics.where_predicates.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<clean::GenericParamDef> for GenericParamDef {
    fn from(generic_param: clean::GenericParamDef) -> Self {
        GenericParamDef { name: generic_param.name.to_string(), kind: generic_param.kind.into() }
    }
}

impl From<clean::GenericParamDefKind> for GenericParamDefKind {
    fn from(kind: clean::GenericParamDefKind) -> Self {
        use clean::GenericParamDefKind::*;
        match kind {
            Lifetime => GenericParamDefKind::Lifetime,
            Type { did: _, bounds, default, synthetic: _ } => GenericParamDefKind::Type {
                bounds: bounds.into_iter().map(Into::into).collect(),
                default: default.map(Into::into),
            },
            Const { did: _, ty } => GenericParamDefKind::Const(ty.into()),
        }
    }
}

impl From<clean::WherePredicate> for WherePredicate {
    fn from(predicate: clean::WherePredicate) -> Self {
        use clean::WherePredicate::*;
        match predicate {
            BoundPredicate { ty, bounds } => WherePredicate::BoundPredicate {
                ty: ty.into(),
                bounds: bounds.into_iter().map(Into::into).collect(),
            },
            RegionPredicate { lifetime, bounds } => WherePredicate::RegionPredicate {
                lifetime: lifetime.0,
                bounds: bounds.into_iter().map(Into::into).collect(),
            },
            EqPredicate { lhs, rhs } => {
                WherePredicate::EqPredicate { lhs: lhs.into(), rhs: rhs.into() }
            }
        }
    }
}

impl From<clean::GenericBound> for GenericBound {
    fn from(bound: clean::GenericBound) -> Self {
        use clean::GenericBound::*;
        match bound {
            TraitBound(clean::PolyTrait { trait_, generic_params }, modifier) => {
                GenericBound::TraitBound {
                    trait_: trait_.into(),
                    generic_params: generic_params.into_iter().map(Into::into).collect(),
                    modifier: modifier.into(),
                }
            }
            Outlives(lifetime) => GenericBound::Outlives(lifetime.0),
        }
    }
}

impl From<rustc_hir::TraitBoundModifier> for TraitBoundModifier {
    fn from(modifier: rustc_hir::TraitBoundModifier) -> Self {
        use rustc_hir::TraitBoundModifier::*;
        match modifier {
            None => TraitBoundModifier::None,
            Maybe => TraitBoundModifier::Maybe,
            MaybeConst => TraitBoundModifier::MaybeConst,
        }
    }
}

impl From<clean::Type> for Type {
    fn from(ty: clean::Type) -> Self {
        use clean::Type::*;
        match ty {
            ResolvedPath { path, param_names, did, is_generic: _ } => Type::ResolvedPath {
                name: path.whole_name(),
                id: did.into(),
                args: path.segments.last().map(|args| Box::new(args.clone().args.into())),
                param_names: param_names
                    .map(|v| v.into_iter().map(Into::into).collect())
                    .unwrap_or_default(),
            },
            Generic(s) => Type::Generic(s.to_string()),
            Primitive(p) => Type::Primitive(p.as_str().to_string()),
            BareFunction(f) => Type::FunctionPointer(Box::new((*f).into())),
            Tuple(t) => Type::Tuple(t.into_iter().map(Into::into).collect()),
            Slice(t) => Type::Slice(Box::new((*t).into())),
            Array(t, s) => Type::Array { type_: Box::new((*t).into()), len: s },
            ImplTrait(g) => Type::ImplTrait(g.into_iter().map(Into::into).collect()),
            Never => Type::Never,
            Infer => Type::Infer,
            RawPointer(mutability, type_) => Type::RawPointer {
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into()),
            },
            BorrowedRef { lifetime, mutability, type_ } => Type::BorrowedRef {
                lifetime: lifetime.map(|l| l.0),
                mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into()),
            },
            QPath { name, self_type, trait_ } => Type::QualifiedPath {
                name: name.to_string(),
                self_type: Box::new((*self_type).into()),
                trait_: Box::new((*trait_).into()),
            },
        }
    }
}

impl From<clean::BareFunctionDecl> for FunctionPointer {
    fn from(bare_decl: clean::BareFunctionDecl) -> Self {
        let clean::BareFunctionDecl { unsafety, generic_params, decl, abi } = bare_decl;
        FunctionPointer {
            is_unsafe: unsafety == rustc_hir::Unsafety::Unsafe,
            generic_params: generic_params.into_iter().map(Into::into).collect(),
            decl: decl.into(),
            abi: abi.to_string(),
        }
    }
}

impl From<clean::FnDecl> for FnDecl {
    fn from(decl: clean::FnDecl) -> Self {
        let clean::FnDecl { inputs, output, c_variadic, attrs: _ } = decl;
        FnDecl {
            inputs: inputs
                .values
                .into_iter()
                .map(|arg| (arg.name.to_string(), arg.type_.into()))
                .collect(),
            output: match output {
                clean::FnRetTy::Return(t) => Some(t.into()),
                clean::FnRetTy::DefaultReturn => None,
            },
            c_variadic,
        }
    }
}

impl From<clean::Trait> for Trait {
    fn from(trait_: clean::Trait) -> Self {
        let clean::Trait { unsafety, items, generics, bounds, is_spotlight: _, is_auto } = trait_;
        Trait {
            is_auto,
            is_unsafe: unsafety == rustc_hir::Unsafety::Unsafe,
            items: ids(items),
            generics: generics.into(),
            bounds: bounds.into_iter().map(Into::into).collect(),
            implementors: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl From<clean::Impl> for Impl {
    fn from(impl_: clean::Impl) -> Self {
        let clean::Impl {
            unsafety,
            generics,
            provided_trait_methods,
            trait_,
            for_,
            items,
            polarity,
            synthetic,
            blanket_impl,
        } = impl_;
        Impl {
            is_unsafe: unsafety == rustc_hir::Unsafety::Unsafe,
            generics: generics.into(),
            provided_trait_methods: provided_trait_methods.into_iter().collect(),
            trait_: trait_.map(Into::into),
            for_: for_.into(),
            items: ids(items),
            negative: polarity == Some(clean::ImplPolarity::Negative),
            synthetic,
            blanket_impl: blanket_impl.map(Into::into),
        }
    }
}

impl From<clean::Function> for Method {
    fn from(function: clean::Function) -> Self {
        let clean::Function { header, decl, generics, all_types: _, ret_types: _ } = function;
        Method {
            decl: decl.into(),
            generics: generics.into(),
            header: stringify_header(&header),
            has_body: true,
        }
    }
}

impl From<clean::Enum> for Enum {
    fn from(enum_: clean::Enum) -> Self {
        let clean::Enum { variants, generics, variants_stripped } = enum_;
        Enum {
            generics: generics.into(),
            variants_stripped,
            variants: ids(variants),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl From<clean::VariantStruct> for Struct {
    fn from(struct_: clean::VariantStruct) -> Self {
        let clean::VariantStruct { struct_type, fields, fields_stripped } = struct_;
        Struct {
            struct_type: struct_type.into(),
            generics: Default::default(),
            fields_stripped,
            fields: ids(fields),
            impls: Vec::new(),
        }
    }
}

impl From<clean::Variant> for Variant {
    fn from(variant: clean::Variant) -> Self {
        use clean::VariantKind::*;
        match variant.kind {
            CLike => Variant::Plain,
            Tuple(t) => Variant::Tuple(t.into_iter().map(Into::into).collect()),
            Struct(s) => Variant::Struct(ids(s.fields)),
        }
    }
}

impl From<clean::Import> for Import {
    fn from(import: clean::Import) -> Self {
        use clean::ImportKind::*;
        match import.kind {
            Simple(s) => Import {
                span: import.source.path.whole_name(),
                name: s,
                id: import.source.did.map(Into::into),
                glob: false,
            },
            Glob => Import {
                span: import.source.path.whole_name(),
                name: import.source.path.last_name().to_string(),
                id: import.source.did.map(Into::into),
                glob: true,
            },
        }
    }
}

impl From<clean::ProcMacro> for ProcMacro {
    fn from(mac: clean::ProcMacro) -> Self {
        ProcMacro { kind: mac.kind.into(), helpers: mac.helpers }
    }
}

impl From<rustc_span::hygiene::MacroKind> for MacroKind {
    fn from(kind: rustc_span::hygiene::MacroKind) -> Self {
        use rustc_span::hygiene::MacroKind::*;
        match kind {
            Bang => MacroKind::Bang,
            Attr => MacroKind::Attr,
            Derive => MacroKind::Derive,
        }
    }
}

impl From<clean::Typedef> for Typedef {
    fn from(typedef: clean::Typedef) -> Self {
        let clean::Typedef { type_, generics, item_type: _ } = typedef;
        Typedef { type_: type_.into(), generics: generics.into() }
    }
}

impl From<clean::OpaqueTy> for OpaqueTy {
    fn from(opaque: clean::OpaqueTy) -> Self {
        OpaqueTy {
            bounds: opaque.bounds.into_iter().map(Into::into).collect(),
            generics: opaque.generics.into(),
        }
    }
}

impl From<clean::Static> for Static {
    fn from(stat: clean::Static) -> Self {
        Static {
            type_: stat.type_.into(),
            mutable: stat.mutability == ast::Mutability::Mut,
            expr: stat.expr,
        }
    }
}

impl From<clean::TraitAlias> for TraitAlias {
    fn from(alias: clean::TraitAlias) -> Self {
        TraitAlias {
            generics: alias.generics.into(),
            params: alias.bounds.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<ItemType> for ItemKind {
    fn from(kind: ItemType) -> Self {
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
    items.into_iter().filter(|x| !x.is_stripped()).map(|i| i.def_id.into()).collect()
}
