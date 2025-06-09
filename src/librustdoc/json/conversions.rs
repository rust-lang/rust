//! These from impls are used to create the JSON types which get serialized. They're very close to
//! the `clean` types but with some fields removed or stringified to simplify the output and not
//! expose unstable compiler internals.

#![allow(rustc::default_hash_types)]

use rustc_abi::ExternAbi;
use rustc_ast::ast;
use rustc_attr_data_structures::{self as attrs, DeprecatedSince};
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::DefId;
use rustc_metadata::rendered_const;
use rustc_middle::{bug, ty};
use rustc_span::{Pos, Symbol, kw};
use rustdoc_json_types::*;

use crate::clean::{self, ItemId};
use crate::formats::FormatRenderer;
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

                (String::from(&**link), self.id_from_item_default(id.into()))
            })
            .collect();
        let docs = item.opt_doc_value();
        let attrs = item.attributes_and_repr(self.tcx, self.cache(), true);
        let span = item.span(self.tcx);
        let visibility = item.visibility(self.tcx);
        let clean::ItemInner { name, item_id, .. } = *item.inner;
        let id = self.id_from_item(&item);
        let inner = match item.kind {
            clean::KeywordItem => return None,
            clean::StrippedItem(ref inner) => {
                match &**inner {
                    // We document stripped modules as with `Module::is_stripped` set to
                    // `true`, to prevent contained items from being orphaned for downstream users,
                    // as JSON does no inlining.
                    clean::ModuleItem(_)
                        if self.imported_items.contains(&item_id.expect_def_id()) =>
                    {
                        from_clean_item(item, self)
                    }
                    _ => return None,
                }
            }
            _ => from_clean_item(item, self),
        };
        Some(Item {
            id,
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
                        begin: (lo.line, lo.col.to_usize() + 1),
                        end: (hi.line, hi.col.to_usize() + 1),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn convert_visibility(&self, v: Option<ty::Visibility<DefId>>) -> Visibility {
        match v {
            None => Visibility::Default,
            Some(ty::Visibility::Public) => Visibility::Public,
            Some(ty::Visibility::Restricted(did)) if did.is_crate_root() => Visibility::Crate,
            Some(ty::Visibility::Restricted(did)) => Visibility::Restricted {
                parent: self.id_from_item_default(did.into()),
                path: self.tcx.def_path(did).to_string_no_crate_verbose(),
            },
        }
    }

    fn ids(&self, items: impl IntoIterator<Item = clean::Item>) -> Vec<Id> {
        items
            .into_iter()
            .filter(|x| !x.is_stripped() && !x.is_keyword())
            .map(|i| self.id_from_item(&i))
            .collect()
    }

    fn ids_keeping_stripped(
        &self,
        items: impl IntoIterator<Item = clean::Item>,
    ) -> Vec<Option<Id>> {
        items
            .into_iter()
            .map(|i| (!i.is_stripped() && !i.is_keyword()).then(|| self.id_from_item(&i)))
            .collect()
    }
}

pub(crate) trait FromClean<T> {
    fn from_clean(f: T, renderer: &JsonRenderer<'_>) -> Self;
}

pub(crate) trait IntoJson<T> {
    fn into_json(self, renderer: &JsonRenderer<'_>) -> T;
}

impl<T, U> IntoJson<U> for T
where
    U: FromClean<T>,
{
    fn into_json(self, renderer: &JsonRenderer<'_>) -> U {
        U::from_clean(self, renderer)
    }
}

impl<I, T, U> FromClean<I> for Vec<U>
where
    I: IntoIterator<Item = T>,
    U: FromClean<T>,
{
    fn from_clean(f: I, renderer: &JsonRenderer<'_>) -> Vec<U> {
        f.into_iter().map(|x| x.into_json(renderer)).collect()
    }
}

pub(crate) fn from_deprecation(deprecation: attrs::Deprecation) -> Deprecation {
    let attrs::Deprecation { since, note, suggestion: _ } = deprecation;
    let since = match since {
        DeprecatedSince::RustcVersion(version) => Some(version.to_string()),
        DeprecatedSince::Future => Some("TBD".to_owned()),
        DeprecatedSince::NonStandard(since) => Some(since.to_string()),
        DeprecatedSince::Unspecified | DeprecatedSince::Err => None,
    };
    Deprecation { since, note: note.map(|s| s.to_string()) }
}

impl FromClean<clean::GenericArgs> for GenericArgs {
    fn from_clean(args: clean::GenericArgs, renderer: &JsonRenderer<'_>) -> Self {
        use clean::GenericArgs::*;
        match args {
            AngleBracketed { args, constraints } => GenericArgs::AngleBracketed {
                args: args.into_json(renderer),
                constraints: constraints.into_json(renderer),
            },
            Parenthesized { inputs, output } => GenericArgs::Parenthesized {
                inputs: inputs.into_json(renderer),
                output: output.map(|a| (*a).into_json(renderer)),
            },
            ReturnTypeNotation => GenericArgs::ReturnTypeNotation,
        }
    }
}

impl FromClean<clean::GenericArg> for GenericArg {
    fn from_clean(arg: clean::GenericArg, renderer: &JsonRenderer<'_>) -> Self {
        use clean::GenericArg::*;
        match arg {
            Lifetime(l) => GenericArg::Lifetime(convert_lifetime(l)),
            Type(t) => GenericArg::Type(t.into_json(renderer)),
            Const(box c) => GenericArg::Const(c.into_json(renderer)),
            Infer => GenericArg::Infer,
        }
    }
}

impl FromClean<clean::Constant> for Constant {
    // FIXME(generic_const_items): Add support for generic const items.
    fn from_clean(constant: clean::Constant, renderer: &JsonRenderer<'_>) -> Self {
        let tcx = renderer.tcx;
        let expr = constant.expr(tcx);
        let value = constant.value(tcx);
        let is_literal = constant.is_literal(tcx);
        Constant { expr, value, is_literal }
    }
}

impl FromClean<clean::ConstantKind> for Constant {
    // FIXME(generic_const_items): Add support for generic const items.
    fn from_clean(constant: clean::ConstantKind, renderer: &JsonRenderer<'_>) -> Self {
        let tcx = renderer.tcx;
        let expr = constant.expr(tcx);
        let value = constant.value(tcx);
        let is_literal = constant.is_literal(tcx);
        Constant { expr, value, is_literal }
    }
}

impl FromClean<clean::AssocItemConstraint> for AssocItemConstraint {
    fn from_clean(constraint: clean::AssocItemConstraint, renderer: &JsonRenderer<'_>) -> Self {
        AssocItemConstraint {
            name: constraint.assoc.name.to_string(),
            args: constraint.assoc.args.into_json(renderer),
            binding: constraint.kind.into_json(renderer),
        }
    }
}

impl FromClean<clean::AssocItemConstraintKind> for AssocItemConstraintKind {
    fn from_clean(kind: clean::AssocItemConstraintKind, renderer: &JsonRenderer<'_>) -> Self {
        use clean::AssocItemConstraintKind::*;
        match kind {
            Equality { term } => AssocItemConstraintKind::Equality(term.into_json(renderer)),
            Bound { bounds } => AssocItemConstraintKind::Constraint(bounds.into_json(renderer)),
        }
    }
}

fn from_clean_item(item: clean::Item, renderer: &JsonRenderer<'_>) -> ItemEnum {
    use clean::ItemKind::*;
    let name = item.name;
    let is_crate = item.is_crate();
    let header = item.fn_header(renderer.tcx);

    match item.inner.kind {
        ModuleItem(m) => {
            ItemEnum::Module(Module { is_crate, items: renderer.ids(m.items), is_stripped: false })
        }
        ImportItem(i) => ItemEnum::Use(i.into_json(renderer)),
        StructItem(s) => ItemEnum::Struct(s.into_json(renderer)),
        UnionItem(u) => ItemEnum::Union(u.into_json(renderer)),
        StructFieldItem(f) => ItemEnum::StructField(f.into_json(renderer)),
        EnumItem(e) => ItemEnum::Enum(e.into_json(renderer)),
        VariantItem(v) => ItemEnum::Variant(v.into_json(renderer)),
        FunctionItem(f) => ItemEnum::Function(from_function(*f, true, header.unwrap(), renderer)),
        ForeignFunctionItem(f, _) => {
            ItemEnum::Function(from_function(*f, false, header.unwrap(), renderer))
        }
        TraitItem(t) => ItemEnum::Trait((*t).into_json(renderer)),
        TraitAliasItem(t) => ItemEnum::TraitAlias(t.into_json(renderer)),
        MethodItem(m, _) => ItemEnum::Function(from_function(*m, true, header.unwrap(), renderer)),
        RequiredMethodItem(m) => {
            ItemEnum::Function(from_function(*m, false, header.unwrap(), renderer))
        }
        ImplItem(i) => ItemEnum::Impl((*i).into_json(renderer)),
        StaticItem(s) => ItemEnum::Static(convert_static(s, rustc_hir::Safety::Safe, renderer)),
        ForeignStaticItem(s, safety) => ItemEnum::Static(convert_static(s, safety, renderer)),
        ForeignTypeItem => ItemEnum::ExternType,
        TypeAliasItem(t) => ItemEnum::TypeAlias(t.into_json(renderer)),
        // FIXME(generic_const_items): Add support for generic free consts
        ConstantItem(ci) => ItemEnum::Constant {
            type_: ci.type_.into_json(renderer),
            const_: ci.kind.into_json(renderer),
        },
        MacroItem(m) => ItemEnum::Macro(m.source),
        ProcMacroItem(m) => ItemEnum::ProcMacro(m.into_json(renderer)),
        PrimitiveItem(p) => {
            ItemEnum::Primitive(Primitive {
                name: p.as_sym().to_string(),
                impls: Vec::new(), // Added in JsonRenderer::item
            })
        }
        // FIXME(generic_const_items): Add support for generic associated consts.
        RequiredAssocConstItem(_generics, ty) => {
            ItemEnum::AssocConst { type_: (*ty).into_json(renderer), value: None }
        }
        // FIXME(generic_const_items): Add support for generic associated consts.
        ProvidedAssocConstItem(ci) | ImplAssocConstItem(ci) => ItemEnum::AssocConst {
            type_: ci.type_.into_json(renderer),
            value: Some(ci.kind.expr(renderer.tcx)),
        },
        RequiredAssocTypeItem(g, b) => ItemEnum::AssocType {
            generics: g.into_json(renderer),
            bounds: b.into_json(renderer),
            type_: None,
        },
        AssocTypeItem(t, b) => ItemEnum::AssocType {
            generics: t.generics.into_json(renderer),
            bounds: b.into_json(renderer),
            type_: Some(t.item_type.unwrap_or(t.type_).into_json(renderer)),
        },
        // `convert_item` early returns `None` for stripped items and keywords.
        KeywordItem => unreachable!(),
        StrippedItem(inner) => {
            match *inner {
                ModuleItem(m) => ItemEnum::Module(Module {
                    is_crate,
                    items: renderer.ids(m.items),
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

impl FromClean<clean::Struct> for Struct {
    fn from_clean(struct_: clean::Struct, renderer: &JsonRenderer<'_>) -> Self {
        let has_stripped_fields = struct_.has_stripped_entries();
        let clean::Struct { ctor_kind, generics, fields } = struct_;

        let kind = match ctor_kind {
            Some(CtorKind::Fn) => StructKind::Tuple(renderer.ids_keeping_stripped(fields)),
            Some(CtorKind::Const) => {
                assert!(fields.is_empty());
                StructKind::Unit
            }
            None => StructKind::Plain { fields: renderer.ids(fields), has_stripped_fields },
        };

        Struct {
            kind,
            generics: generics.into_json(renderer),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromClean<clean::Union> for Union {
    fn from_clean(union_: clean::Union, renderer: &JsonRenderer<'_>) -> Self {
        let has_stripped_fields = union_.has_stripped_entries();
        let clean::Union { generics, fields } = union_;
        Union {
            generics: generics.into_json(renderer),
            has_stripped_fields,
            fields: renderer.ids(fields),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

pub(crate) fn from_fn_header(header: &rustc_hir::FnHeader) -> FunctionHeader {
    FunctionHeader {
        is_async: header.is_async(),
        is_const: header.is_const(),
        is_unsafe: header.is_unsafe(),
        abi: convert_abi(header.abi),
    }
}

fn convert_abi(a: ExternAbi) -> Abi {
    match a {
        ExternAbi::Rust => Abi::Rust,
        ExternAbi::C { unwind } => Abi::C { unwind },
        ExternAbi::Cdecl { unwind } => Abi::Cdecl { unwind },
        ExternAbi::Stdcall { unwind } => Abi::Stdcall { unwind },
        ExternAbi::Fastcall { unwind } => Abi::Fastcall { unwind },
        ExternAbi::Aapcs { unwind } => Abi::Aapcs { unwind },
        ExternAbi::Win64 { unwind } => Abi::Win64 { unwind },
        ExternAbi::SysV64 { unwind } => Abi::SysV64 { unwind },
        ExternAbi::System { unwind } => Abi::System { unwind },
        _ => Abi::Other(a.to_string()),
    }
}

fn convert_lifetime(l: clean::Lifetime) -> String {
    l.0.to_string()
}

impl FromClean<clean::Generics> for Generics {
    fn from_clean(generics: clean::Generics, renderer: &JsonRenderer<'_>) -> Self {
        Generics {
            params: generics.params.into_json(renderer),
            where_predicates: generics.where_predicates.into_json(renderer),
        }
    }
}

impl FromClean<clean::GenericParamDef> for GenericParamDef {
    fn from_clean(generic_param: clean::GenericParamDef, renderer: &JsonRenderer<'_>) -> Self {
        GenericParamDef {
            name: generic_param.name.to_string(),
            kind: generic_param.kind.into_json(renderer),
        }
    }
}

impl FromClean<clean::GenericParamDefKind> for GenericParamDefKind {
    fn from_clean(kind: clean::GenericParamDefKind, renderer: &JsonRenderer<'_>) -> Self {
        use clean::GenericParamDefKind::*;
        match kind {
            Lifetime { outlives } => GenericParamDefKind::Lifetime {
                outlives: outlives.into_iter().map(convert_lifetime).collect(),
            },
            Type { bounds, default, synthetic } => GenericParamDefKind::Type {
                bounds: bounds.into_json(renderer),
                default: default.map(|x| (*x).into_json(renderer)),
                is_synthetic: synthetic,
            },
            Const { ty, default, synthetic: _ } => GenericParamDefKind::Const {
                type_: (*ty).into_json(renderer),
                default: default.map(|x| *x),
            },
        }
    }
}

impl FromClean<clean::WherePredicate> for WherePredicate {
    fn from_clean(predicate: clean::WherePredicate, renderer: &JsonRenderer<'_>) -> Self {
        use clean::WherePredicate::*;
        match predicate {
            BoundPredicate { ty, bounds, bound_params } => WherePredicate::BoundPredicate {
                type_: ty.into_json(renderer),
                bounds: bounds.into_json(renderer),
                generic_params: bound_params
                    .into_iter()
                    .map(|x| {
                        let name = x.name.to_string();
                        let kind = match x.kind {
                            clean::GenericParamDefKind::Lifetime { outlives } => {
                                GenericParamDefKind::Lifetime {
                                    outlives: outlives.iter().map(|lt| lt.0.to_string()).collect(),
                                }
                            }
                            clean::GenericParamDefKind::Type { bounds, default, synthetic } => {
                                GenericParamDefKind::Type {
                                    bounds: bounds
                                        .into_iter()
                                        .map(|bound| bound.into_json(renderer))
                                        .collect(),
                                    default: default.map(|ty| (*ty).into_json(renderer)),
                                    is_synthetic: synthetic,
                                }
                            }
                            clean::GenericParamDefKind::Const { ty, default, synthetic: _ } => {
                                GenericParamDefKind::Const {
                                    type_: (*ty).into_json(renderer),
                                    default: default.map(|d| *d),
                                }
                            }
                        };
                        GenericParamDef { name, kind }
                    })
                    .collect(),
            },
            RegionPredicate { lifetime, bounds } => WherePredicate::LifetimePredicate {
                lifetime: convert_lifetime(lifetime),
                outlives: bounds
                    .iter()
                    .map(|bound| match bound {
                        clean::GenericBound::Outlives(lt) => convert_lifetime(*lt),
                        _ => bug!("found non-outlives-bound on lifetime predicate"),
                    })
                    .collect(),
            },
            EqPredicate { lhs, rhs } => WherePredicate::EqPredicate {
                // The LHS currently has type `Type` but it should be a `QualifiedPath` since it may
                // refer to an associated const. However, `EqPredicate` shouldn't exist in the first
                // place: <https://github.com/rust-lang/rust/141368>.
                lhs: lhs.into_json(renderer),
                rhs: rhs.into_json(renderer),
            },
        }
    }
}

impl FromClean<clean::GenericBound> for GenericBound {
    fn from_clean(bound: clean::GenericBound, renderer: &JsonRenderer<'_>) -> Self {
        use clean::GenericBound::*;
        match bound {
            TraitBound(clean::PolyTrait { trait_, generic_params }, modifier) => {
                GenericBound::TraitBound {
                    trait_: trait_.into_json(renderer),
                    generic_params: generic_params.into_json(renderer),
                    modifier: from_trait_bound_modifier(modifier),
                }
            }
            Outlives(lifetime) => GenericBound::Outlives(convert_lifetime(lifetime)),
            Use(args) => GenericBound::Use(
                args.iter()
                    .map(|arg| match arg {
                        clean::PreciseCapturingArg::Lifetime(lt) => {
                            PreciseCapturingArg::Lifetime(convert_lifetime(*lt))
                        }
                        clean::PreciseCapturingArg::Param(param) => {
                            PreciseCapturingArg::Param(param.to_string())
                        }
                    })
                    .collect(),
            ),
        }
    }
}

pub(crate) fn from_trait_bound_modifier(
    modifiers: rustc_hir::TraitBoundModifiers,
) -> TraitBoundModifier {
    use rustc_hir as hir;
    let hir::TraitBoundModifiers { constness, polarity } = modifiers;
    match (constness, polarity) {
        (hir::BoundConstness::Never, hir::BoundPolarity::Positive) => TraitBoundModifier::None,
        (hir::BoundConstness::Never, hir::BoundPolarity::Maybe(_)) => TraitBoundModifier::Maybe,
        (hir::BoundConstness::Maybe(_), hir::BoundPolarity::Positive) => {
            TraitBoundModifier::MaybeConst
        }
        // FIXME: Fill out the rest of this matrix.
        _ => TraitBoundModifier::None,
    }
}

impl FromClean<clean::Type> for Type {
    fn from_clean(ty: clean::Type, renderer: &JsonRenderer<'_>) -> Self {
        use clean::Type::{
            Array, BareFunction, BorrowedRef, Generic, ImplTrait, Infer, Primitive, QPath,
            RawPointer, SelfTy, Slice, Tuple, UnsafeBinder,
        };

        match ty {
            clean::Type::Path { path } => Type::ResolvedPath(path.into_json(renderer)),
            clean::Type::DynTrait(bounds, lt) => Type::DynTrait(DynTrait {
                lifetime: lt.map(convert_lifetime),
                traits: bounds.into_json(renderer),
            }),
            Generic(s) => Type::Generic(s.to_string()),
            // FIXME: add dedicated variant to json Type?
            SelfTy => Type::Generic("Self".to_owned()),
            Primitive(p) => Type::Primitive(p.as_sym().to_string()),
            BareFunction(f) => Type::FunctionPointer(Box::new((*f).into_json(renderer))),
            Tuple(t) => Type::Tuple(t.into_json(renderer)),
            Slice(t) => Type::Slice(Box::new((*t).into_json(renderer))),
            Array(t, s) => {
                Type::Array { type_: Box::new((*t).into_json(renderer)), len: s.to_string() }
            }
            clean::Type::Pat(t, p) => Type::Pat {
                type_: Box::new((*t).into_json(renderer)),
                __pat_unstable_do_not_use: p.to_string(),
            },
            ImplTrait(g) => Type::ImplTrait(g.into_json(renderer)),
            Infer => Type::Infer,
            RawPointer(mutability, type_) => Type::RawPointer {
                is_mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_json(renderer)),
            },
            BorrowedRef { lifetime, mutability, type_ } => Type::BorrowedRef {
                lifetime: lifetime.map(convert_lifetime),
                is_mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_json(renderer)),
            },
            QPath(qpath) => (*qpath).into_json(renderer),
            // FIXME(unsafe_binder): Implement rustdoc-json.
            UnsafeBinder(_) => todo!(),
        }
    }
}

impl FromClean<clean::Path> for Path {
    fn from_clean(path: clean::Path, renderer: &JsonRenderer<'_>) -> Path {
        Path {
            path: path.whole_name(),
            id: renderer.id_from_item_default(path.def_id().into()),
            args: path.segments.last().map(|args| Box::new(args.clone().args.into_json(renderer))),
        }
    }
}

impl FromClean<clean::QPathData> for Type {
    fn from_clean(qpath: clean::QPathData, renderer: &JsonRenderer<'_>) -> Self {
        let clean::QPathData { assoc, self_type, should_fully_qualify: _, trait_ } = qpath;

        Self::QualifiedPath {
            name: assoc.name.to_string(),
            args: Box::new(assoc.args.into_json(renderer)),
            self_type: Box::new(self_type.into_json(renderer)),
            trait_: trait_.map(|trait_| trait_.into_json(renderer)),
        }
    }
}

impl FromClean<clean::Term> for Term {
    fn from_clean(term: clean::Term, renderer: &JsonRenderer<'_>) -> Term {
        match term {
            clean::Term::Type(ty) => Term::Type(ty.into_json(renderer)),
            clean::Term::Constant(c) => Term::Constant(c.into_json(renderer)),
        }
    }
}

impl FromClean<clean::BareFunctionDecl> for FunctionPointer {
    fn from_clean(bare_decl: clean::BareFunctionDecl, renderer: &JsonRenderer<'_>) -> Self {
        let clean::BareFunctionDecl { safety, generic_params, decl, abi } = bare_decl;
        FunctionPointer {
            header: FunctionHeader {
                is_unsafe: safety.is_unsafe(),
                is_const: false,
                is_async: false,
                abi: convert_abi(abi),
            },
            generic_params: generic_params.into_json(renderer),
            sig: decl.into_json(renderer),
        }
    }
}

impl FromClean<clean::FnDecl> for FunctionSignature {
    fn from_clean(decl: clean::FnDecl, renderer: &JsonRenderer<'_>) -> Self {
        let clean::FnDecl { inputs, output, c_variadic } = decl;
        FunctionSignature {
            inputs: inputs
                .into_iter()
                .map(|param| {
                    // `_` is the most sensible name for missing param names.
                    let name = param.name.unwrap_or(kw::Underscore).to_string();
                    let type_ = param.type_.into_json(renderer);
                    (name, type_)
                })
                .collect(),
            output: if output.is_unit() { None } else { Some(output.into_json(renderer)) },
            is_c_variadic: c_variadic,
        }
    }
}

impl FromClean<clean::Trait> for Trait {
    fn from_clean(trait_: clean::Trait, renderer: &JsonRenderer<'_>) -> Self {
        let tcx = renderer.tcx;
        let is_auto = trait_.is_auto(tcx);
        let is_unsafe = trait_.safety(tcx).is_unsafe();
        let is_dyn_compatible = trait_.is_dyn_compatible(tcx);
        let clean::Trait { items, generics, bounds, .. } = trait_;
        Trait {
            is_auto,
            is_unsafe,
            is_dyn_compatible,
            items: renderer.ids(items),
            generics: generics.into_json(renderer),
            bounds: bounds.into_json(renderer),
            implementations: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromClean<clean::PolyTrait> for PolyTrait {
    fn from_clean(
        clean::PolyTrait { trait_, generic_params }: clean::PolyTrait,
        renderer: &JsonRenderer<'_>,
    ) -> Self {
        PolyTrait {
            trait_: trait_.into_json(renderer),
            generic_params: generic_params.into_json(renderer),
        }
    }
}

impl FromClean<clean::Impl> for Impl {
    fn from_clean(impl_: clean::Impl, renderer: &JsonRenderer<'_>) -> Self {
        let provided_trait_methods = impl_.provided_trait_methods(renderer.tcx);
        let clean::Impl { safety, generics, trait_, for_, items, polarity, kind } = impl_;
        // FIXME: use something like ImplKind in JSON?
        let (is_synthetic, blanket_impl) = match kind {
            clean::ImplKind::Normal | clean::ImplKind::FakeVariadic => (false, None),
            clean::ImplKind::Auto => (true, None),
            clean::ImplKind::Blanket(ty) => (false, Some(*ty)),
        };
        let is_negative = match polarity {
            ty::ImplPolarity::Positive | ty::ImplPolarity::Reservation => false,
            ty::ImplPolarity::Negative => true,
        };
        Impl {
            is_unsafe: safety.is_unsafe(),
            generics: generics.into_json(renderer),
            provided_trait_methods: provided_trait_methods
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            trait_: trait_.map(|path| path.into_json(renderer)),
            for_: for_.into_json(renderer),
            items: renderer.ids(items),
            is_negative,
            is_synthetic,
            blanket_impl: blanket_impl.map(|x| x.into_json(renderer)),
        }
    }
}

pub(crate) fn from_function(
    clean::Function { decl, generics }: clean::Function,
    has_body: bool,
    header: rustc_hir::FnHeader,
    renderer: &JsonRenderer<'_>,
) -> Function {
    Function {
        sig: decl.into_json(renderer),
        generics: generics.into_json(renderer),
        header: from_fn_header(&header),
        has_body,
    }
}

impl FromClean<clean::Enum> for Enum {
    fn from_clean(enum_: clean::Enum, renderer: &JsonRenderer<'_>) -> Self {
        let has_stripped_variants = enum_.has_stripped_entries();
        let clean::Enum { variants, generics } = enum_;
        Enum {
            generics: generics.into_json(renderer),
            has_stripped_variants,
            variants: renderer.ids(variants),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromClean<clean::Variant> for Variant {
    fn from_clean(variant: clean::Variant, renderer: &JsonRenderer<'_>) -> Self {
        use clean::VariantKind::*;

        let discriminant = variant.discriminant.map(|d| d.into_json(renderer));

        let kind = match variant.kind {
            CLike => VariantKind::Plain,
            Tuple(fields) => VariantKind::Tuple(renderer.ids_keeping_stripped(fields)),
            Struct(s) => VariantKind::Struct {
                has_stripped_fields: s.has_stripped_entries(),
                fields: renderer.ids(s.fields),
            },
        };

        Variant { kind, discriminant }
    }
}

impl FromClean<clean::Discriminant> for Discriminant {
    fn from_clean(disr: clean::Discriminant, renderer: &JsonRenderer<'_>) -> Self {
        let tcx = renderer.tcx;
        Discriminant {
            // expr is only none if going through the inlining path, which gets
            // `rustc_middle` types, not `rustc_hir`, but because JSON never inlines
            // the expr is always some.
            expr: disr.expr(tcx).unwrap(),
            value: disr.value(tcx, false),
        }
    }
}

impl FromClean<clean::Import> for Use {
    fn from_clean(import: clean::Import, renderer: &JsonRenderer<'_>) -> Self {
        use clean::ImportKind::*;
        let (name, is_glob) = match import.kind {
            Simple(s) => (s.to_string(), false),
            Glob => (
                import.source.path.last_opt().unwrap_or_else(|| Symbol::intern("*")).to_string(),
                true,
            ),
        };
        Use {
            source: import.source.path.whole_name(),
            name,
            id: import.source.did.map(ItemId::from).map(|i| renderer.id_from_item_default(i)),
            is_glob,
        }
    }
}

impl FromClean<clean::ProcMacro> for ProcMacro {
    fn from_clean(mac: clean::ProcMacro, _renderer: &JsonRenderer<'_>) -> Self {
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

impl FromClean<Box<clean::TypeAlias>> for TypeAlias {
    fn from_clean(type_alias: Box<clean::TypeAlias>, renderer: &JsonRenderer<'_>) -> Self {
        let clean::TypeAlias { type_, generics, item_type: _, inner_type: _ } = *type_alias;
        TypeAlias { type_: type_.into_json(renderer), generics: generics.into_json(renderer) }
    }
}

fn convert_static(
    stat: clean::Static,
    safety: rustc_hir::Safety,
    renderer: &JsonRenderer<'_>,
) -> Static {
    let tcx = renderer.tcx;
    Static {
        type_: (*stat.type_).into_json(renderer),
        is_mutable: stat.mutability == ast::Mutability::Mut,
        is_unsafe: safety.is_unsafe(),
        expr: stat
            .expr
            .map(|e| rendered_const(tcx, tcx.hir_body(e), tcx.hir_body_owner_def_id(e)))
            .unwrap_or_default(),
    }
}

impl FromClean<clean::TraitAlias> for TraitAlias {
    fn from_clean(alias: clean::TraitAlias, renderer: &JsonRenderer<'_>) -> Self {
        TraitAlias {
            generics: alias.generics.into_json(renderer),
            params: alias.bounds.into_json(renderer),
        }
    }
}

impl FromClean<ItemType> for ItemKind {
    fn from_clean(kind: ItemType, _renderer: &JsonRenderer<'_>) -> Self {
        use ItemType::*;
        match kind {
            Module => ItemKind::Module,
            ExternCrate => ItemKind::ExternCrate,
            Import => ItemKind::Use,
            Struct => ItemKind::Struct,
            Union => ItemKind::Union,
            Enum => ItemKind::Enum,
            Function | TyMethod | Method => ItemKind::Function,
            TypeAlias => ItemKind::TypeAlias,
            Static => ItemKind::Static,
            Constant => ItemKind::Constant,
            Trait => ItemKind::Trait,
            Impl => ItemKind::Impl,
            StructField => ItemKind::StructField,
            Variant => ItemKind::Variant,
            Macro => ItemKind::Macro,
            Primitive => ItemKind::Primitive,
            AssocConst => ItemKind::AssocConst,
            AssocType => ItemKind::AssocType,
            ForeignType => ItemKind::ExternType,
            Keyword => ItemKind::Keyword,
            TraitAlias => ItemKind::TraitAlias,
            ProcAttribute => ItemKind::ProcAttribute,
            ProcDerive => ItemKind::ProcDerive,
        }
    }
}
