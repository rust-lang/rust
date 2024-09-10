//! These from impls are used to create the JSON types which get serialized. They're very close to
//! the `clean` types but with some fields removed or stringified to simplify the output and not
//! expose unstable compiler internals.

#![allow(rustc::default_hash_types)]

use std::fmt;

use rustc_ast::ast;
use rustc_attr::DeprecatedSince;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::DefId;
use rustc_metadata::rendered_const;
use rustc_middle::bug;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::symbol::sym;
use rustc_span::{Pos, Symbol};
use rustc_target::spec::abi::Abi as RustcAbi;
use rustdoc_json_types::*;

use crate::clean::{self, ItemId};
use crate::formats::item_type::ItemType;
use crate::formats::FormatRenderer;
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

                (String::from(&**link), id_from_item_default(id.into(), self.tcx))
            })
            .collect();
        let docs = item.opt_doc_value();
        let attrs = item.attributes(self.tcx, self.cache(), true);
        let span = item.span(self.tcx);
        let visibility = item.visibility(self.tcx);
        let clean::Item { name, item_id, .. } = item;
        let id = id_from_item(&item, self.tcx);
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
                        from_clean_item(item, self.tcx)
                    }
                    _ => return None,
                }
            }
            _ => from_clean_item(item, self.tcx),
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

    fn convert_visibility(&self, v: Option<ty::Visibility<DefId>>) -> Visibility {
        match v {
            None => Visibility::Default,
            Some(ty::Visibility::Public) => Visibility::Public,
            Some(ty::Visibility::Restricted(did)) if did.is_crate_root() => Visibility::Crate,
            Some(ty::Visibility::Restricted(did)) => Visibility::Restricted {
                parent: id_from_item_default(did.into(), self.tcx),
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
    let rustc_attr::Deprecation { since, note, suggestion: _ } = deprecation;
    let since = match since {
        DeprecatedSince::RustcVersion(version) => Some(version.to_string()),
        DeprecatedSince::Future => Some("TBD".to_owned()),
        DeprecatedSince::NonStandard(since) => Some(since.to_string()),
        DeprecatedSince::Unspecified | DeprecatedSince::Err => None,
    };
    Deprecation { since, note: note.map(|s| s.to_string()) }
}

impl FromWithTcx<clean::GenericArgs> for GenericArgs {
    fn from_tcx(args: clean::GenericArgs, tcx: TyCtxt<'_>) -> Self {
        use clean::GenericArgs::*;
        match args {
            AngleBracketed { args, constraints } => GenericArgs::AngleBracketed {
                args: args.into_vec().into_tcx(tcx),
                constraints: constraints.into_tcx(tcx),
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
    // FIXME(generic_const_items): Add support for generic const items.
    fn from_tcx(constant: clean::Constant, tcx: TyCtxt<'_>) -> Self {
        let expr = constant.expr(tcx);
        let value = constant.value(tcx);
        let is_literal = constant.is_literal(tcx);
        Constant { expr, value, is_literal }
    }
}

impl FromWithTcx<clean::ConstantKind> for Constant {
    // FIXME(generic_const_items): Add support for generic const items.
    fn from_tcx(constant: clean::ConstantKind, tcx: TyCtxt<'_>) -> Self {
        let expr = constant.expr(tcx);
        let value = constant.value(tcx);
        let is_literal = constant.is_literal(tcx);
        Constant { expr, value, is_literal }
    }
}

impl FromWithTcx<clean::AssocItemConstraint> for AssocItemConstraint {
    fn from_tcx(constraint: clean::AssocItemConstraint, tcx: TyCtxt<'_>) -> Self {
        AssocItemConstraint {
            name: constraint.assoc.name.to_string(),
            args: constraint.assoc.args.into_tcx(tcx),
            binding: constraint.kind.into_tcx(tcx),
        }
    }
}

impl FromWithTcx<clean::AssocItemConstraintKind> for AssocItemConstraintKind {
    fn from_tcx(kind: clean::AssocItemConstraintKind, tcx: TyCtxt<'_>) -> Self {
        use clean::AssocItemConstraintKind::*;
        match kind {
            Equality { term } => AssocItemConstraintKind::Equality(term.into_tcx(tcx)),
            Bound { bounds } => AssocItemConstraintKind::Constraint(bounds.into_tcx(tcx)),
        }
    }
}

#[inline]
pub(crate) fn id_from_item_default(item_id: ItemId, tcx: TyCtxt<'_>) -> Id {
    id_from_item_inner(item_id, tcx, None, None)
}

/// It generates an ID as follows:
///
/// `CRATE_ID:ITEM_ID[:NAME_ID][-EXTRA]`:
///   * If there is no `name`, `NAME_ID` is not generated.
///   * If there is no `extra`, `EXTRA` is not generated.
///
/// * `name` is the item's name if available (it's not for impl blocks for example).
/// * `extra` is used for reexports: it contains the ID of the reexported item. It is used to allow
///   to have items with the same name but different types to both appear in the generated JSON.
pub(crate) fn id_from_item_inner(
    item_id: ItemId,
    tcx: TyCtxt<'_>,
    name: Option<Symbol>,
    extra: Option<&Id>,
) -> Id {
    struct DisplayDefId<'a, 'b>(DefId, TyCtxt<'a>, Option<&'b Id>, Option<Symbol>);

    impl<'a, 'b> fmt::Display for DisplayDefId<'a, 'b> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let DisplayDefId(def_id, tcx, extra, name) = self;
            // We need this workaround because primitive types' DefId actually refers to
            // their parent module, which isn't present in the output JSON items. So
            // instead, we directly get the primitive symbol and convert it to u32 to
            // generate the ID.
            let s;
            let extra = if let Some(e) = extra {
                s = format!("-{}", e.0);
                &s
            } else {
                ""
            };
            let name = match name {
                Some(name) => format!(":{}", name.as_u32()),
                None => {
                    // We need this workaround because primitive types' DefId actually refers to
                    // their parent module, which isn't present in the output JSON items. So
                    // instead, we directly get the primitive symbol and convert it to u32 to
                    // generate the ID.
                    if matches!(tcx.def_kind(def_id), DefKind::Mod)
                        && let Some(prim) = tcx
                            .get_attrs(*def_id, sym::rustc_doc_primitive)
                            .find_map(|attr| attr.value_str())
                    {
                        format!(":{}", prim.as_u32())
                    } else {
                        tcx.opt_item_name(*def_id)
                            .map(|n| format!(":{}", n.as_u32()))
                            .unwrap_or_default()
                    }
                }
            };
            write!(f, "{}:{}{name}{extra}", def_id.krate.as_u32(), u32::from(def_id.index))
        }
    }

    match item_id {
        ItemId::DefId(did) => Id(format!("{}", DisplayDefId(did, tcx, extra, name))),
        ItemId::Blanket { for_, impl_id } => Id(format!(
            "b:{}-{}",
            DisplayDefId(impl_id, tcx, None, None),
            DisplayDefId(for_, tcx, extra, name)
        )),
        ItemId::Auto { for_, trait_ } => Id(format!(
            "a:{}-{}",
            DisplayDefId(trait_, tcx, None, None),
            DisplayDefId(for_, tcx, extra, name)
        )),
    }
}

pub(crate) fn id_from_item(item: &clean::Item, tcx: TyCtxt<'_>) -> Id {
    match item.kind {
        clean::ItemKind::ImportItem(ref import) => {
            let extra =
                import.source.did.map(ItemId::from).map(|i| id_from_item_inner(i, tcx, None, None));
            id_from_item_inner(item.item_id, tcx, item.name, extra.as_ref())
        }
        _ => id_from_item_inner(item.item_id, tcx, item.name, None),
    }
}

fn from_clean_item(item: clean::Item, tcx: TyCtxt<'_>) -> ItemEnum {
    use clean::ItemKind::*;
    let name = item.name;
    let is_crate = item.is_crate();
    let header = item.fn_header(tcx);

    match item.inner.kind {
        ModuleItem(m) => {
            ItemEnum::Module(Module { is_crate, items: ids(m.items, tcx), is_stripped: false })
        }
        ImportItem(i) => ItemEnum::Use(i.into_tcx(tcx)),
        StructItem(s) => ItemEnum::Struct(s.into_tcx(tcx)),
        UnionItem(u) => ItemEnum::Union(u.into_tcx(tcx)),
        StructFieldItem(f) => ItemEnum::StructField(f.into_tcx(tcx)),
        EnumItem(e) => ItemEnum::Enum(e.into_tcx(tcx)),
        VariantItem(v) => ItemEnum::Variant(v.into_tcx(tcx)),
        FunctionItem(f) => ItemEnum::Function(from_function(f, true, header.unwrap(), tcx)),
        ForeignFunctionItem(f, _) => {
            ItemEnum::Function(from_function(f, false, header.unwrap(), tcx))
        }
        TraitItem(t) => ItemEnum::Trait((*t).into_tcx(tcx)),
        TraitAliasItem(t) => ItemEnum::TraitAlias(t.into_tcx(tcx)),
        MethodItem(m, _) => ItemEnum::Function(from_function(m, true, header.unwrap(), tcx)),
        TyMethodItem(m) => ItemEnum::Function(from_function(m, false, header.unwrap(), tcx)),
        ImplItem(i) => ItemEnum::Impl((*i).into_tcx(tcx)),
        StaticItem(s) => ItemEnum::Static(s.into_tcx(tcx)),
        ForeignStaticItem(s, _) => ItemEnum::Static(s.into_tcx(tcx)),
        ForeignTypeItem => ItemEnum::ExternType,
        TypeAliasItem(t) => ItemEnum::TypeAlias(t.into_tcx(tcx)),
        // FIXME(generic_const_items): Add support for generic free consts
        ConstantItem(ci) => {
            ItemEnum::Constant { type_: ci.type_.into_tcx(tcx), const_: ci.kind.into_tcx(tcx) }
        }
        MacroItem(m) => ItemEnum::Macro(m.source),
        ProcMacroItem(m) => ItemEnum::ProcMacro(m.into_tcx(tcx)),
        PrimitiveItem(p) => {
            ItemEnum::Primitive(Primitive {
                name: p.as_sym().to_string(),
                impls: Vec::new(), // Added in JsonRenderer::item
            })
        }
        // FIXME(generic_const_items): Add support for generic associated consts.
        TyAssocConstItem(_generics, ty) => {
            ItemEnum::AssocConst { type_: (*ty).into_tcx(tcx), value: None }
        }
        // FIXME(generic_const_items): Add support for generic associated consts.
        AssocConstItem(ci) => {
            ItemEnum::AssocConst { type_: ci.type_.into_tcx(tcx), value: Some(ci.kind.expr(tcx)) }
        }
        TyAssocTypeItem(g, b) => {
            ItemEnum::AssocType { generics: g.into_tcx(tcx), bounds: b.into_tcx(tcx), type_: None }
        }
        AssocTypeItem(t, b) => ItemEnum::AssocType {
            generics: t.generics.into_tcx(tcx),
            bounds: b.into_tcx(tcx),
            type_: Some(t.item_type.unwrap_or(t.type_).into_tcx(tcx)),
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
        let has_stripped_fields = struct_.has_stripped_entries();
        let clean::Struct { ctor_kind, generics, fields } = struct_;

        let kind = match ctor_kind {
            Some(CtorKind::Fn) => StructKind::Tuple(ids_keeping_stripped(fields, tcx)),
            Some(CtorKind::Const) => {
                assert!(fields.is_empty());
                StructKind::Unit
            }
            None => StructKind::Plain { fields: ids(fields, tcx), has_stripped_fields },
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
        let has_stripped_fields = union_.has_stripped_entries();
        let clean::Union { generics, fields } = union_;
        Union {
            generics: generics.into_tcx(tcx),
            has_stripped_fields,
            fields: ids(fields, tcx),
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
            Type { bounds, default, synthetic } => GenericParamDefKind::Type {
                bounds: bounds.into_tcx(tcx),
                default: default.map(|x| (*x).into_tcx(tcx)),
                is_synthetic: synthetic,
            },
            Const { ty, default, synthetic: _ } => GenericParamDefKind::Const {
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
                                        .map(|bound| bound.into_tcx(tcx))
                                        .collect(),
                                    default: default.map(|ty| (*ty).into_tcx(tcx)),
                                    is_synthetic: synthetic,
                                }
                            }
                            clean::GenericParamDefKind::Const { ty, default, synthetic: _ } => {
                                GenericParamDefKind::Const {
                                    type_: (*ty).into_tcx(tcx),
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
            Use(args) => GenericBound::Use(args.into_iter().map(|arg| arg.to_string()).collect()),
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
        // FIXME(const_trait_impl): Create rjt::TBM::Const and map to it once always-const bounds
        // are less experimental.
        Const => TraitBoundModifier::None,
        // FIXME(negative-bounds): This bound should be rendered negative, but
        // since that's experimental, maybe let's not add it to the rustdoc json
        // API just now...
        Negative => TraitBoundModifier::None,
    }
}

impl FromWithTcx<clean::Type> for Type {
    fn from_tcx(ty: clean::Type, tcx: TyCtxt<'_>) -> Self {
        use clean::Type::{
            Array, BareFunction, BorrowedRef, Generic, ImplTrait, Infer, Primitive, QPath,
            RawPointer, SelfTy, Slice, Tuple,
        };

        match ty {
            clean::Type::Path { path } => Type::ResolvedPath(path.into_tcx(tcx)),
            clean::Type::DynTrait(bounds, lt) => Type::DynTrait(DynTrait {
                lifetime: lt.map(convert_lifetime),
                traits: bounds.into_tcx(tcx),
            }),
            Generic(s) => Type::Generic(s.to_string()),
            // FIXME: add dedicated variant to json Type?
            SelfTy => Type::Generic("Self".to_owned()),
            Primitive(p) => Type::Primitive(p.as_sym().to_string()),
            BareFunction(f) => Type::FunctionPointer(Box::new((*f).into_tcx(tcx))),
            Tuple(t) => Type::Tuple(t.into_tcx(tcx)),
            Slice(t) => Type::Slice(Box::new((*t).into_tcx(tcx))),
            Array(t, s) => Type::Array { type_: Box::new((*t).into_tcx(tcx)), len: s.to_string() },
            clean::Type::Pat(t, p) => Type::Pat {
                type_: Box::new((*t).into_tcx(tcx)),
                __pat_unstable_do_not_use: p.to_string(),
            },
            ImplTrait(g) => Type::ImplTrait(g.into_tcx(tcx)),
            Infer => Type::Infer,
            RawPointer(mutability, type_) => Type::RawPointer {
                is_mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_tcx(tcx)),
            },
            BorrowedRef { lifetime, mutability, type_ } => Type::BorrowedRef {
                lifetime: lifetime.map(convert_lifetime),
                is_mutable: mutability == ast::Mutability::Mut,
                type_: Box::new((*type_).into_tcx(tcx)),
            },
            QPath(box clean::QPathData { assoc, self_type, trait_, .. }) => Type::QualifiedPath {
                name: assoc.name.to_string(),
                args: Box::new(assoc.args.into_tcx(tcx)),
                self_type: Box::new(self_type.into_tcx(tcx)),
                trait_: trait_.map(|trait_| trait_.into_tcx(tcx)),
            },
        }
    }
}

impl FromWithTcx<clean::Path> for Path {
    fn from_tcx(path: clean::Path, tcx: TyCtxt<'_>) -> Path {
        Path {
            name: path.whole_name(),
            id: id_from_item_default(path.def_id().into(), tcx),
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
        let clean::BareFunctionDecl { safety, generic_params, decl, abi } = bare_decl;
        FunctionPointer {
            header: FunctionHeader {
                is_unsafe: matches!(safety, rustc_hir::Safety::Unsafe),
                is_const: false,
                is_async: false,
                abi: convert_abi(abi),
            },
            generic_params: generic_params.into_tcx(tcx),
            sig: decl.into_tcx(tcx),
        }
    }
}

impl FromWithTcx<clean::FnDecl> for FunctionSignature {
    fn from_tcx(decl: clean::FnDecl, tcx: TyCtxt<'_>) -> Self {
        let clean::FnDecl { inputs, output, c_variadic } = decl;
        FunctionSignature {
            inputs: inputs
                .values
                .into_iter()
                .map(|arg| (arg.name.to_string(), arg.type_.into_tcx(tcx)))
                .collect(),
            output: if output.is_unit() { None } else { Some(output.into_tcx(tcx)) },
            is_c_variadic: c_variadic,
        }
    }
}

impl FromWithTcx<clean::Trait> for Trait {
    fn from_tcx(trait_: clean::Trait, tcx: TyCtxt<'_>) -> Self {
        let is_auto = trait_.is_auto(tcx);
        let is_unsafe = trait_.safety(tcx) == rustc_hir::Safety::Unsafe;
        let is_object_safe = trait_.is_object_safe(tcx);
        let clean::Trait { items, generics, bounds, .. } = trait_;
        Trait {
            is_auto,
            is_unsafe,
            is_object_safe,
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
            is_unsafe: safety == rustc_hir::Safety::Unsafe,
            generics: generics.into_tcx(tcx),
            provided_trait_methods: provided_trait_methods
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            trait_: trait_.map(|path| path.into_tcx(tcx)),
            for_: for_.into_tcx(tcx),
            items: ids(items, tcx),
            is_negative,
            is_synthetic,
            blanket_impl: blanket_impl.map(|x| x.into_tcx(tcx)),
        }
    }
}

pub(crate) fn from_function(
    function: Box<clean::Function>,
    has_body: bool,
    header: rustc_hir::FnHeader,
    tcx: TyCtxt<'_>,
) -> Function {
    let clean::Function { decl, generics } = *function;
    Function {
        sig: decl.into_tcx(tcx),
        generics: generics.into_tcx(tcx),
        header: from_fn_header(&header),
        has_body,
    }
}

impl FromWithTcx<clean::Enum> for Enum {
    fn from_tcx(enum_: clean::Enum, tcx: TyCtxt<'_>) -> Self {
        let has_stripped_variants = enum_.has_stripped_entries();
        let clean::Enum { variants, generics } = enum_;
        Enum {
            generics: generics.into_tcx(tcx),
            has_stripped_variants,
            variants: ids(variants, tcx),
            impls: Vec::new(), // Added in JsonRenderer::item
        }
    }
}

impl FromWithTcx<clean::Variant> for Variant {
    fn from_tcx(variant: clean::Variant, tcx: TyCtxt<'_>) -> Self {
        use clean::VariantKind::*;

        let discriminant = variant.discriminant.map(|d| d.into_tcx(tcx));

        let kind = match variant.kind {
            CLike => VariantKind::Plain,
            Tuple(fields) => VariantKind::Tuple(ids_keeping_stripped(fields, tcx)),
            Struct(s) => VariantKind::Struct {
                has_stripped_fields: s.has_stripped_entries(),
                fields: ids(s.fields, tcx),
            },
        };

        Variant { kind, discriminant }
    }
}

impl FromWithTcx<clean::Discriminant> for Discriminant {
    fn from_tcx(disr: clean::Discriminant, tcx: TyCtxt<'_>) -> Self {
        Discriminant {
            // expr is only none if going through the inlining path, which gets
            // `rustc_middle` types, not `rustc_hir`, but because JSON never inlines
            // the expr is always some.
            expr: disr.expr(tcx).unwrap(),
            value: disr.value(tcx, false),
        }
    }
}

impl FromWithTcx<clean::Import> for Use {
    fn from_tcx(import: clean::Import, tcx: TyCtxt<'_>) -> Self {
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
            id: import.source.did.map(ItemId::from).map(|i| id_from_item_default(i, tcx)),
            is_glob,
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

impl FromWithTcx<Box<clean::TypeAlias>> for TypeAlias {
    fn from_tcx(type_alias: Box<clean::TypeAlias>, tcx: TyCtxt<'_>) -> Self {
        let clean::TypeAlias { type_, generics, item_type: _, inner_type: _ } = *type_alias;
        TypeAlias { type_: type_.into_tcx(tcx), generics: generics.into_tcx(tcx) }
    }
}

impl FromWithTcx<clean::Static> for Static {
    fn from_tcx(stat: clean::Static, tcx: TyCtxt<'_>) -> Self {
        Static {
            type_: (*stat.type_).into_tcx(tcx),
            is_mutable: stat.mutability == ast::Mutability::Mut,
            expr: stat
                .expr
                .map(|e| rendered_const(tcx, tcx.hir().body(e), tcx.hir().body_owner_def_id(e)))
                .unwrap_or_default(),
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

fn ids(items: impl IntoIterator<Item = clean::Item>, tcx: TyCtxt<'_>) -> Vec<Id> {
    items
        .into_iter()
        .filter(|x| !x.is_stripped() && !x.is_keyword())
        .map(|i| id_from_item(&i, tcx))
        .collect()
}

fn ids_keeping_stripped(
    items: impl IntoIterator<Item = clean::Item>,
    tcx: TyCtxt<'_>,
) -> Vec<Option<Id>> {
    items
        .into_iter()
        .map(
            |i| {
                if !i.is_stripped() && !i.is_keyword() { Some(id_from_item(&i, tcx)) } else { None }
            },
        )
        .collect()
}
