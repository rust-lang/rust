//! Attributes for anything that is not name resolution.
//!
//! The fundamental idea of this module stems from the observation that most "interesting"
//! attributes have a more memory-compact form than storing their full syntax, and
//! that most of the attributes are flags, and those that are not are rare. Therefore,
//! this module defines [`AttrFlags`], which is a bitflag enum that contains only a yes/no
//! answer to whether an attribute is present on an item. For most attributes, that's all
//! that is interesting us; for the rest of them, we define another query that extracts
//! their data. A key part is that every one of those queries will have a wrapper method
//! that queries (or is given) the `AttrFlags` and checks for the presence of the attribute;
//! if it is not present, we do not call the query, to prevent Salsa from needing to record
//! its value. This way, queries are only called on items that have the attribute, which is
//! usually only a few.
//!
//! Documentation (doc comments and `#[doc = "..."]` attributes) is handled by the [`docs`]
//! submodule.

use std::{convert::Infallible, iter::Peekable, ops::ControlFlow};

use base_db::Crate;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    InFile, Lookup,
    attrs::{AstKeyValueMetaExt, AstPathExt, expand_cfg_attr},
};
use intern::Symbol;
use itertools::Itertools;
use la_arena::ArenaMap;
use rustc_abi::ReprOptions;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use syntax::{
    AstNode, AstToken, NodeOrToken, SmolStr, SourceFile, T,
    ast::{self, HasAttrs, TokenTreeChildren},
};
use tt::TextSize;

use crate::{
    AdtId, AstIdLoc, AttrDefId, FieldId, FunctionId, GenericDefId, HasModule, LifetimeParamId,
    LocalFieldId, MacroId, ModuleId, TypeOrConstParamId, VariantId,
    db::DefDatabase,
    hir::generics::{GenericParams, LocalLifetimeParamId, LocalTypeOrConstParamId},
    nameres::ModuleOrigin,
    resolver::{HasResolver, Resolver},
    src::{HasChildSource, HasSource},
};

pub mod docs;

pub use self::docs::{Docs, IsInnerDoc};

#[inline]
fn attrs_from_ast_id_loc<N: AstNode + Into<ast::AnyHasAttrs>>(
    db: &dyn DefDatabase,
    lookup: impl Lookup<Database = dyn DefDatabase, Data = impl AstIdLoc<Ast = N> + HasModule>,
) -> (InFile<ast::AnyHasAttrs>, Crate) {
    let loc = lookup.lookup(db);
    let source = loc.source(db);
    let krate = loc.krate(db);
    (source.map(|it| it.into()), krate)
}

#[inline]
fn extract_doc_tt_attr(attr_flags: &mut AttrFlags, tt: ast::TokenTree) {
    for atom in DocAtom::parse(tt) {
        match atom {
            DocAtom::Flag(flag) => match &*flag {
                "notable_trait" => attr_flags.insert(AttrFlags::IS_DOC_NOTABLE_TRAIT),
                "hidden" => attr_flags.insert(AttrFlags::IS_DOC_HIDDEN),
                _ => {}
            },
            DocAtom::KeyValue { key, value: _ } => match &*key {
                "alias" => attr_flags.insert(AttrFlags::HAS_DOC_ALIASES),
                "keyword" => attr_flags.insert(AttrFlags::HAS_DOC_KEYWORD),
                _ => {}
            },
            DocAtom::Alias(_) => attr_flags.insert(AttrFlags::HAS_DOC_ALIASES),
        }
    }
}

fn extract_ra_completions(attr_flags: &mut AttrFlags, tt: ast::TokenTree) {
    let tt = TokenTreeChildren::new(&tt);
    if let Ok(NodeOrToken::Token(option)) = Itertools::exactly_one(tt)
        && option.kind().is_any_identifier()
    {
        match option.text() {
            "ignore_flyimport" => attr_flags.insert(AttrFlags::COMPLETE_IGNORE_FLYIMPORT),
            "ignore_methods" => attr_flags.insert(AttrFlags::COMPLETE_IGNORE_METHODS),
            "ignore_flyimport_methods" => {
                attr_flags.insert(AttrFlags::COMPLETE_IGNORE_FLYIMPORT_METHODS)
            }
            _ => {}
        }
    }
}

fn extract_ra_macro_style(attr_flags: &mut AttrFlags, tt: ast::TokenTree) {
    let tt = TokenTreeChildren::new(&tt);
    if let Ok(NodeOrToken::Token(option)) = Itertools::exactly_one(tt)
        && option.kind().is_any_identifier()
    {
        match option.text() {
            "braces" => attr_flags.insert(AttrFlags::MACRO_STYLE_BRACES),
            "brackets" => attr_flags.insert(AttrFlags::MACRO_STYLE_BRACKETS),
            "parentheses" => attr_flags.insert(AttrFlags::MACRO_STYLE_PARENTHESES),
            _ => {}
        }
    }
}

fn extract_rustc_skip_during_method_dispatch(attr_flags: &mut AttrFlags, tt: ast::TokenTree) {
    let iter = TokenTreeChildren::new(&tt);
    for kind in iter {
        if let NodeOrToken::Token(kind) = kind
            && kind.kind().is_any_identifier()
        {
            match kind.text() {
                "array" => attr_flags.insert(AttrFlags::RUSTC_SKIP_ARRAY_DURING_METHOD_DISPATCH),
                "boxed_slice" => {
                    attr_flags.insert(AttrFlags::RUSTC_SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH)
                }
                _ => {}
            }
        }
    }
}

#[inline]
fn match_attr_flags(attr_flags: &mut AttrFlags, attr: ast::Meta) -> ControlFlow<Infallible> {
    match attr {
        ast::Meta::CfgMeta(_) => attr_flags.insert(AttrFlags::HAS_CFG),
        ast::Meta::KeyValueMeta(attr) => {
            let Some(key) = attr.path().as_one_segment() else { return ControlFlow::Continue(()) };
            match &*key {
                "deprecated" => attr_flags.insert(AttrFlags::IS_DEPRECATED),
                "ignore" => attr_flags.insert(AttrFlags::IS_IGNORE),
                "lang" => attr_flags.insert(AttrFlags::LANG_ITEM),
                "path" => attr_flags.insert(AttrFlags::HAS_PATH),
                "unstable" => attr_flags.insert(AttrFlags::IS_UNSTABLE),
                "export_name" => {
                    if let Some(value) = attr.value_string()
                        && *value == *"main"
                    {
                        attr_flags.insert(AttrFlags::IS_EXPORT_NAME_MAIN);
                    }
                }
                _ => {}
            }
        }
        ast::Meta::TokenTreeMeta(attr) => {
            let (Some((first_segment, second_segment)), Some(tt)) =
                (attr.path().as_up_to_two_segment(), attr.token_tree())
            else {
                return ControlFlow::Continue(());
            };
            match second_segment {
                None => match &*first_segment {
                    "deprecated" => attr_flags.insert(AttrFlags::IS_DEPRECATED),
                    "doc" => extract_doc_tt_attr(attr_flags, tt),
                    "repr" | "rustc_scalable_vector" => attr_flags.insert(AttrFlags::HAS_REPR),
                    "target_feature" => attr_flags.insert(AttrFlags::HAS_TARGET_FEATURE),
                    "proc_macro_derive" | "rustc_builtin_macro" => {
                        attr_flags.insert(AttrFlags::IS_DERIVE_OR_BUILTIN_MACRO)
                    }
                    "unstable" => attr_flags.insert(AttrFlags::IS_UNSTABLE),
                    "rustc_layout_scalar_valid_range_start"
                    | "rustc_layout_scalar_valid_range_end" => {
                        attr_flags.insert(AttrFlags::RUSTC_LAYOUT_SCALAR_VALID_RANGE)
                    }
                    "rustc_legacy_const_generics" => {
                        attr_flags.insert(AttrFlags::HAS_LEGACY_CONST_GENERICS)
                    }
                    "rustc_skip_during_method_dispatch" => {
                        extract_rustc_skip_during_method_dispatch(attr_flags, tt)
                    }
                    "rustc_deprecated_safe_2024" => {
                        attr_flags.insert(AttrFlags::RUSTC_DEPRECATED_SAFE_2024)
                    }
                    _ => {}
                },
                Some(second_segment) => match &*first_segment {
                    "rust_analyzer" => match &*second_segment {
                        "completions" => extract_ra_completions(attr_flags, tt),
                        "macro_style" => extract_ra_macro_style(attr_flags, tt),
                        _ => {}
                    },
                    _ => {}
                },
            }
        }
        ast::Meta::PathMeta(attr) => {
            let is_test = attr.path().is_some_and(|path| {
                let Some(segment1) = (|| path.segment()?.name_ref())() else { return false };
                let segment2 = path.qualifier();
                let segment3 = segment2.as_ref().and_then(|it| it.qualifier());
                let segment4 = segment3.as_ref().and_then(|it| it.qualifier());
                let segment3 = segment3.and_then(|it| it.segment()?.name_ref());
                let segment4 = segment4.and_then(|it| it.segment()?.name_ref());
                segment1.text() == "test"
                    && segment3.is_none_or(|it| it.text() == "prelude")
                    && segment4.is_none_or(|it| matches!(&*it.text(), "core" | "std"))
            });
            if is_test {
                attr_flags.insert(AttrFlags::IS_TEST);
            }

            let Some((first_segment, second_segment)) = attr.path().as_up_to_two_segment() else {
                return ControlFlow::Continue(());
            };
            match second_segment {
                None => match &*first_segment {
                    "rustc_has_incoherent_inherent_impls" => {
                        attr_flags.insert(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS)
                    }
                    "rustc_allow_incoherent_impl" => {
                        attr_flags.insert(AttrFlags::RUSTC_ALLOW_INCOHERENT_IMPL)
                    }
                    "rustc_scalable_vector" => attr_flags.insert(AttrFlags::HAS_REPR),
                    "fundamental" => attr_flags.insert(AttrFlags::FUNDAMENTAL),
                    "no_std" => attr_flags.insert(AttrFlags::IS_NO_STD),
                    "may_dangle" => attr_flags.insert(AttrFlags::MAY_DANGLE),
                    "rustc_paren_sugar" => attr_flags.insert(AttrFlags::RUSTC_PAREN_SUGAR),
                    "rustc_coinductive" => attr_flags.insert(AttrFlags::RUSTC_COINDUCTIVE),
                    "rustc_force_inline" => attr_flags.insert(AttrFlags::RUSTC_FORCE_INLINE),
                    "unstable" => attr_flags.insert(AttrFlags::IS_UNSTABLE),
                    "deprecated" => attr_flags.insert(AttrFlags::IS_DEPRECATED),
                    "macro_export" => attr_flags.insert(AttrFlags::IS_MACRO_EXPORT),
                    "no_mangle" => attr_flags.insert(AttrFlags::NO_MANGLE),
                    "pointee" => attr_flags.insert(AttrFlags::IS_POINTEE),
                    "non_exhaustive" => attr_flags.insert(AttrFlags::NON_EXHAUSTIVE),
                    "ignore" => attr_flags.insert(AttrFlags::IS_IGNORE),
                    "bench" => attr_flags.insert(AttrFlags::IS_BENCH),
                    "rustc_const_panic_str" => attr_flags.insert(AttrFlags::RUSTC_CONST_PANIC_STR),
                    "rustc_intrinsic" => attr_flags.insert(AttrFlags::RUSTC_INTRINSIC),
                    "rustc_safe_intrinsic" => attr_flags.insert(AttrFlags::RUSTC_SAFE_INTRINSIC),
                    "rustc_intrinsic_must_be_overridden" => {
                        attr_flags.insert(AttrFlags::RUSTC_INTRINSIC_MUST_BE_OVERRIDDEN)
                    }
                    "rustc_allocator" => attr_flags.insert(AttrFlags::RUSTC_ALLOCATOR),
                    "rustc_deallocator" => attr_flags.insert(AttrFlags::RUSTC_DEALLOCATOR),
                    "rustc_reallocator" => attr_flags.insert(AttrFlags::RUSTC_REALLOCATOR),
                    "rustc_allocator_zeroed" => {
                        attr_flags.insert(AttrFlags::RUSTC_ALLOCATOR_ZEROED)
                    }
                    "rustc_reservation_impl" => {
                        attr_flags.insert(AttrFlags::RUSTC_RESERVATION_IMPL)
                    }
                    "rustc_deprecated_safe_2024" => {
                        attr_flags.insert(AttrFlags::RUSTC_DEPRECATED_SAFE_2024)
                    }
                    "rustc_skip_array_during_method_dispatch" => {
                        attr_flags.insert(AttrFlags::RUSTC_SKIP_ARRAY_DURING_METHOD_DISPATCH)
                    }
                    _ => {}
                },
                Some(second_segment) => match &*first_segment {
                    "rust_analyzer" => match &*second_segment {
                        "skip" => attr_flags.insert(AttrFlags::RUST_ANALYZER_SKIP),
                        "prefer_underscore_import" => {
                            attr_flags.insert(AttrFlags::PREFER_UNDERSCORE_IMPORT)
                        }
                        _ => {}
                    },
                    _ => {}
                },
            }
        }
        _ => {}
    };
    ControlFlow::Continue(())
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct AttrFlags: u64 {
        const RUST_ANALYZER_SKIP = 1 << 0;

        const LANG_ITEM = 1 << 1;

        const HAS_DOC_ALIASES = 1 << 2;
        const HAS_DOC_KEYWORD = 1 << 3;
        const IS_DOC_NOTABLE_TRAIT = 1 << 4;
        const IS_DOC_HIDDEN = 1 << 5;

        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS = 1 << 6;
        const RUSTC_ALLOW_INCOHERENT_IMPL = 1 << 7;
        const FUNDAMENTAL = 1 << 8;
        const RUSTC_SKIP_ARRAY_DURING_METHOD_DISPATCH = 1 << 9;
        const RUSTC_SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH = 1 << 10;
        const HAS_REPR = 1 << 11;
        const HAS_TARGET_FEATURE = 1 << 12;
        const RUSTC_DEPRECATED_SAFE_2024 = 1 << 13;
        const HAS_LEGACY_CONST_GENERICS = 1 << 14;
        const NO_MANGLE = 1 << 15;
        const NON_EXHAUSTIVE = 1 << 16;
        const RUSTC_RESERVATION_IMPL = 1 << 17;
        const RUSTC_CONST_PANIC_STR = 1 << 18;
        const MAY_DANGLE = 1 << 19;

        const RUSTC_INTRINSIC = 1 << 20;
        const RUSTC_SAFE_INTRINSIC = 1 << 21;
        const RUSTC_INTRINSIC_MUST_BE_OVERRIDDEN = 1 << 22;
        const RUSTC_ALLOCATOR = 1 << 23;
        const RUSTC_DEALLOCATOR = 1 << 24;
        const RUSTC_REALLOCATOR = 1 << 25;
        const RUSTC_ALLOCATOR_ZEROED = 1 << 26;

        const IS_UNSTABLE = 1 << 27;
        const IS_IGNORE = 1 << 28;
        // FIXME: `IS_TEST` and `IS_BENCH` should be based on semantic information, not textual match.
        const IS_BENCH = 1 << 29;
        const IS_TEST = 1 << 30;
        const IS_EXPORT_NAME_MAIN = 1 << 31;
        const IS_MACRO_EXPORT = 1 << 32;
        const IS_NO_STD = 1 << 33;
        const IS_DERIVE_OR_BUILTIN_MACRO = 1 << 34;
        const IS_DEPRECATED = 1 << 35;
        const HAS_PATH = 1 << 36;
        const HAS_CFG = 1 << 37;

        const COMPLETE_IGNORE_FLYIMPORT = 1 << 38;
        const COMPLETE_IGNORE_FLYIMPORT_METHODS = 1 << 39;
        const COMPLETE_IGNORE_METHODS = 1 << 40;

        const RUSTC_LAYOUT_SCALAR_VALID_RANGE = 1 << 41;
        const RUSTC_PAREN_SUGAR = 1 << 42;
        const RUSTC_COINDUCTIVE = 1 << 43;
        const RUSTC_FORCE_INLINE = 1 << 44;
        const IS_POINTEE = 1 << 45;

        const MACRO_STYLE_BRACES = 1 << 46;
        const MACRO_STYLE_BRACKETS = 1 << 47;
        const MACRO_STYLE_PARENTHESES = 1 << 48;

        const PREFER_UNDERSCORE_IMPORT = 1 << 49;
    }
}

pub fn parse_extra_crate_attrs(db: &dyn DefDatabase, krate: Crate) -> Option<SourceFile> {
    let crate_data = krate.data(db);
    let crate_attrs = &crate_data.crate_attrs;
    if crate_attrs.is_empty() {
        return None;
    }
    // All attributes are already enclosed in `#![]`.
    let combined = crate_attrs.concat();
    let p = SourceFile::parse(&combined, crate_data.edition);

    let errs = p.errors();
    if !errs.is_empty() {
        let base_msg = "Failed to parse extra crate-level attribute";
        let crate_name =
            krate.extra_data(db).display_name.as_ref().map_or("{unknown}", |name| name.as_str());
        let mut errs = errs.iter().peekable();
        let mut offset = TextSize::from(0);
        for raw_attr in crate_attrs {
            let attr_end = offset + TextSize::of(&**raw_attr);
            if errs.peeking_take_while(|e| e.range().start() < attr_end).count() > 0 {
                tracing::error!("{base_msg} {raw_attr} for crate {crate_name}");
            }
            offset = attr_end
        }
        return None;
    }

    Some(p.tree())
}

fn attrs_source(
    db: &dyn DefDatabase,
    owner: AttrDefId,
) -> (InFile<ast::AnyHasAttrs>, Option<InFile<ast::Module>>, Option<SourceFile>, Crate) {
    let (owner, krate) = match owner {
        AttrDefId::ModuleId(id) => {
            let def_map = id.def_map(db);
            let krate = def_map.krate();
            let (definition, declaration, extra_crate_attrs) = match def_map[id].origin {
                ModuleOrigin::CrateRoot { definition } => {
                    let definition_source = definition.parse(db).tree();
                    let definition = InFile::new(definition.into(), definition_source.into());
                    let extra_crate_attrs = parse_extra_crate_attrs(db, krate);
                    (definition, None, extra_crate_attrs)
                }
                ModuleOrigin::File { declaration, declaration_tree_id, definition, .. } => {
                    let definition_source = definition.parse(db).tree();
                    let definition = InFile::new(definition.into(), definition_source.into());
                    let declaration = InFile::new(declaration_tree_id.file_id(), declaration);
                    let declaration = declaration.with_value(declaration.to_node(db));
                    (definition, Some(declaration), None)
                }
                ModuleOrigin::Inline { definition_tree_id, definition } => {
                    let definition = InFile::new(definition_tree_id.file_id(), definition);
                    let definition = definition.with_value(definition.to_node(db).into());
                    (definition, None, None)
                }
                ModuleOrigin::BlockExpr { block, .. } => {
                    let definition = block.to_node(db);
                    (block.with_value(definition.into()), None, None)
                }
            };
            return (definition, declaration, extra_crate_attrs, krate);
        }
        AttrDefId::AdtId(AdtId::StructId(it)) => attrs_from_ast_id_loc(db, it),
        AttrDefId::AdtId(AdtId::UnionId(it)) => attrs_from_ast_id_loc(db, it),
        AttrDefId::AdtId(AdtId::EnumId(it)) => attrs_from_ast_id_loc(db, it),
        AttrDefId::FunctionId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::EnumVariantId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::StaticId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::ConstId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::TraitId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::TypeAliasId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::MacroId(MacroId::MacroRulesId(it)) => attrs_from_ast_id_loc(db, it),
        AttrDefId::MacroId(MacroId::Macro2Id(it)) => attrs_from_ast_id_loc(db, it),
        AttrDefId::MacroId(MacroId::ProcMacroId(it)) => attrs_from_ast_id_loc(db, it),
        AttrDefId::ImplId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::ExternBlockId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::ExternCrateId(it) => attrs_from_ast_id_loc(db, it),
        AttrDefId::UseId(it) => attrs_from_ast_id_loc(db, it),
    };
    (owner, None, None, krate)
}

fn resolver_for_attr_def_id(db: &dyn DefDatabase, owner: AttrDefId) -> Resolver<'_> {
    match owner {
        AttrDefId::ModuleId(id) => id.resolver(db),
        AttrDefId::AdtId(AdtId::StructId(id)) => id.resolver(db),
        AttrDefId::AdtId(AdtId::UnionId(id)) => id.resolver(db),
        AttrDefId::AdtId(AdtId::EnumId(id)) => id.resolver(db),
        AttrDefId::FunctionId(id) => id.resolver(db),
        AttrDefId::EnumVariantId(id) => id.resolver(db),
        AttrDefId::StaticId(id) => id.resolver(db),
        AttrDefId::ConstId(id) => id.resolver(db),
        AttrDefId::TraitId(id) => id.resolver(db),
        AttrDefId::TypeAliasId(id) => id.resolver(db),
        AttrDefId::MacroId(MacroId::Macro2Id(id)) => id.resolver(db),
        AttrDefId::MacroId(MacroId::MacroRulesId(id)) => id.resolver(db),
        AttrDefId::MacroId(MacroId::ProcMacroId(id)) => id.resolver(db),
        AttrDefId::ImplId(id) => id.resolver(db),
        AttrDefId::ExternBlockId(id) => id.resolver(db),
        AttrDefId::ExternCrateId(id) => id.resolver(db),
        AttrDefId::UseId(id) => id.resolver(db),
    }
}

fn collect_attrs<BreakValue>(
    db: &dyn DefDatabase,
    owner: AttrDefId,
    mut callback: impl FnMut(ast::Meta) -> ControlFlow<BreakValue>,
) -> Option<BreakValue> {
    let (source, outer_mod_decl, extra_crate_attrs, krate) = attrs_source(db, owner);
    let extra_attrs = extra_crate_attrs
        .into_iter()
        .flat_map(|src| src.attrs())
        .chain(outer_mod_decl.into_iter().flat_map(|it| it.value.attrs()));

    let mut cfg_options = None;
    expand_cfg_attr(
        extra_attrs.chain(ast::attrs_including_inner(&source.value)),
        || cfg_options.get_or_insert_with(|| krate.cfg_options(db)),
        move |meta, _| callback(meta),
    )
}

fn collect_field_attrs<T>(
    db: &dyn DefDatabase,
    variant: VariantId,
    mut field_attrs: impl FnMut(&CfgOptions, InFile<ast::AnyHasAttrs>) -> T,
) -> ArenaMap<LocalFieldId, T> {
    let (variant_syntax, krate) = match variant {
        VariantId::EnumVariantId(it) => attrs_from_ast_id_loc(db, it),
        VariantId::StructId(it) => attrs_from_ast_id_loc(db, it),
        VariantId::UnionId(it) => attrs_from_ast_id_loc(db, it),
    };
    let cfg_options = krate.cfg_options(db);
    let variant_syntax = variant_syntax
        .with_value(ast::VariantDef::cast(variant_syntax.value.syntax().clone()).unwrap());
    let fields = match &variant_syntax.value {
        ast::VariantDef::Struct(it) => it.field_list(),
        ast::VariantDef::Union(it) => it.record_field_list().map(ast::FieldList::RecordFieldList),
        ast::VariantDef::Variant(it) => it.field_list(),
    };
    let Some(fields) = fields else {
        return ArenaMap::new();
    };

    let mut result = ArenaMap::new();
    let mut idx = 0;
    match fields {
        ast::FieldList::RecordFieldList(fields) => {
            for field in fields.fields() {
                if AttrFlags::is_cfg_enabled_for(&field, cfg_options).is_ok() {
                    result.insert(
                        la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(idx)),
                        field_attrs(cfg_options, variant_syntax.with_value(field.into())),
                    );
                    idx += 1;
                }
            }
        }
        ast::FieldList::TupleFieldList(fields) => {
            for field in fields.fields() {
                if AttrFlags::is_cfg_enabled_for(&field, cfg_options).is_ok() {
                    result.insert(
                        la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(idx)),
                        field_attrs(cfg_options, variant_syntax.with_value(field.into())),
                    );
                    idx += 1;
                }
            }
        }
    }
    result.shrink_to_fit();
    result
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RustcLayoutScalarValidRange {
    pub start: Option<u128>,
    pub end: Option<u128>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DeriveInfo {
    pub trait_name: Symbol,
    pub helpers: Box<[Symbol]>,
}

fn extract_doc_aliases(result: &mut Vec<Symbol>, attr: ast::Meta) -> ControlFlow<Infallible> {
    if let ast::Meta::TokenTreeMeta(attr) = attr
        && attr.path().is1("doc")
        && let Some(tt) = attr.token_tree()
    {
        for atom in DocAtom::parse(tt) {
            match atom {
                DocAtom::Alias(aliases) => {
                    result.extend(aliases.into_iter().map(|alias| Symbol::intern(&alias)))
                }
                DocAtom::KeyValue { key, value } if key == "alias" => {
                    result.push(Symbol::intern(&value))
                }
                _ => {}
            }
        }
    }
    ControlFlow::Continue(())
}

fn extract_cfgs(result: &mut Vec<CfgExpr>, attr: ast::Meta) -> ControlFlow<Infallible> {
    if let ast::Meta::CfgMeta(attr) = attr
        && let Some(cfg_predicate) = attr.cfg_predicate()
    {
        result.push(CfgExpr::parse_from_ast(cfg_predicate));
    }
    ControlFlow::Continue(())
}

#[salsa::tracked]
impl AttrFlags {
    #[salsa::tracked]
    pub fn query(db: &dyn DefDatabase, owner: AttrDefId) -> AttrFlags {
        let mut attr_flags = AttrFlags::empty();
        collect_attrs(db, owner, |attr| match_attr_flags(&mut attr_flags, attr));
        attr_flags
    }

    #[inline]
    pub fn query_field(db: &dyn DefDatabase, field: FieldId) -> AttrFlags {
        return field_attr_flags(db, field.parent)
            .get(field.local_id)
            .copied()
            .unwrap_or_else(AttrFlags::empty);

        #[salsa::tracked(returns(ref))]
        fn field_attr_flags(
            db: &dyn DefDatabase,
            variant: VariantId,
        ) -> ArenaMap<LocalFieldId, AttrFlags> {
            collect_field_attrs(db, variant, |cfg_options, field| {
                let mut attr_flags = AttrFlags::empty();
                expand_cfg_attr(
                    field.value.attrs(),
                    || cfg_options,
                    |attr, _| match_attr_flags(&mut attr_flags, attr),
                );
                attr_flags
            })
        }
    }

    #[inline]
    pub fn query_generic_params(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> &(ArenaMap<LocalLifetimeParamId, AttrFlags>, ArenaMap<LocalTypeOrConstParamId, AttrFlags>)
    {
        let generic_params = GenericParams::of(db, def);
        let params_count_excluding_self =
            generic_params.len() - usize::from(generic_params.trait_self_param().is_some());
        if params_count_excluding_self == 0 {
            return const { &(ArenaMap::new(), ArenaMap::new()) };
        }
        return generic_params_attr_flags(db, def);

        #[salsa::tracked(returns(ref))]
        fn generic_params_attr_flags(
            db: &dyn DefDatabase,
            def: GenericDefId,
        ) -> (ArenaMap<LocalLifetimeParamId, AttrFlags>, ArenaMap<LocalTypeOrConstParamId, AttrFlags>)
        {
            let mut lifetimes = ArenaMap::new();
            let mut type_and_consts = ArenaMap::new();

            let mut cfg_options = None;
            let mut cfg_options =
                || *cfg_options.get_or_insert_with(|| def.krate(db).cfg_options(db));

            let lifetimes_source = HasChildSource::<LocalLifetimeParamId>::child_source(&def, db);
            for (lifetime_id, lifetime) in lifetimes_source.value.iter() {
                let mut attr_flags = AttrFlags::empty();
                expand_cfg_attr(lifetime.attrs(), &mut cfg_options, |attr, _| {
                    match_attr_flags(&mut attr_flags, attr)
                });
                if !attr_flags.is_empty() {
                    lifetimes.insert(lifetime_id, attr_flags);
                }
            }

            let type_and_consts_source =
                HasChildSource::<LocalTypeOrConstParamId>::child_source(&def, db);
            for (type_or_const_id, type_or_const) in type_and_consts_source.value.iter() {
                let mut attr_flags = AttrFlags::empty();
                expand_cfg_attr(type_or_const.attrs(), &mut cfg_options, |attr, _| {
                    match_attr_flags(&mut attr_flags, attr)
                });
                if !attr_flags.is_empty() {
                    type_and_consts.insert(type_or_const_id, attr_flags);
                }
            }

            lifetimes.shrink_to_fit();
            type_and_consts.shrink_to_fit();
            (lifetimes, type_and_consts)
        }
    }

    #[inline]
    pub fn query_lifetime_param(db: &dyn DefDatabase, owner: LifetimeParamId) -> AttrFlags {
        AttrFlags::query_generic_params(db, owner.parent)
            .0
            .get(owner.local_id)
            .copied()
            .unwrap_or_else(AttrFlags::empty)
    }
    #[inline]
    pub fn query_type_or_const_param(db: &dyn DefDatabase, owner: TypeOrConstParamId) -> AttrFlags {
        AttrFlags::query_generic_params(db, owner.parent)
            .1
            .get(owner.local_id)
            .copied()
            .unwrap_or_else(AttrFlags::empty)
    }

    pub(crate) fn is_cfg_enabled_for(
        owner: &dyn HasAttrs,
        cfg_options: &CfgOptions,
    ) -> Result<(), CfgExpr> {
        let attrs = ast::attrs_including_inner(owner);
        let result = expand_cfg_attr(
            attrs,
            || cfg_options,
            |attr, _| {
                if let ast::Meta::CfgMeta(attr) = attr
                    && let Some(cfg_predicate) = attr.cfg_predicate()
                    && let cfg = CfgExpr::parse_from_ast(cfg_predicate)
                    && cfg_options.check(&cfg) == Some(false)
                {
                    ControlFlow::Break(cfg)
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        match result {
            Some(cfg) => Err(cfg),
            None => Ok(()),
        }
    }

    #[inline]
    pub fn lang_item(db: &dyn DefDatabase, owner: AttrDefId) -> Option<Symbol> {
        AttrFlags::query(db, owner).lang_item_with_attrs(db, owner)
    }

    #[inline]
    pub fn lang_item_with_attrs(self, db: &dyn DefDatabase, owner: AttrDefId) -> Option<Symbol> {
        if !self.contains(AttrFlags::LANG_ITEM) {
            // Don't create the query in case this is not a lang item, this wastes memory.
            return None;
        }

        return lang_item(db, owner);

        #[salsa::tracked]
        fn lang_item(db: &dyn DefDatabase, owner: AttrDefId) -> Option<Symbol> {
            collect_attrs(db, owner, |attr| {
                if let ast::Meta::KeyValueMeta(attr) = attr
                    && attr.path().is1("lang")
                    && let Some(value) = attr.value_string()
                {
                    ControlFlow::Break(Symbol::intern(&value))
                } else {
                    ControlFlow::Continue(())
                }
            })
        }
    }

    #[inline]
    pub fn repr(db: &dyn DefDatabase, owner: AdtId) -> Option<ReprOptions> {
        if !AttrFlags::query(db, owner.into()).contains(AttrFlags::HAS_REPR) {
            // Don't create the query in case this has no repr, this wastes memory.
            return None;
        }

        return repr(db, owner);

        #[salsa::tracked]
        fn repr(db: &dyn DefDatabase, owner: AdtId) -> Option<ReprOptions> {
            let mut result = None;
            collect_attrs::<Infallible>(db, owner.into(), |attr| {
                let mut current = None;
                if let ast::Meta::TokenTreeMeta(attr) = &attr
                    && let Some(path) = attr.path()
                    && let Some(tt) = attr.token_tree()
                {
                    if path.is1("repr")
                        && let Some(repr) = parse_repr_tt(&tt)
                    {
                        current = Some(repr);
                    } else if path.is1("rustc_scalable_vector")
                        && let mut tt = TokenTreeChildren::new(&tt)
                        && let Some(NodeOrToken::Token(scalable)) = tt.next()
                        && let Some(scalable) = ast::IntNumber::cast(scalable)
                        && let Ok(scalable) = scalable.value()
                        && let Ok(scalable) = scalable.try_into()
                    {
                        current = Some(ReprOptions {
                            scalable: Some(rustc_abi::ScalableElt::ElementCount(scalable)),
                            ..ReprOptions::default()
                        });
                    }
                } else if let ast::Meta::PathMeta(attr) = &attr
                    && attr.path().is1("rustc_scalable_vector")
                {
                    current = Some(ReprOptions {
                        scalable: Some(rustc_abi::ScalableElt::Container),
                        ..ReprOptions::default()
                    });
                }

                if let Some(current) = current {
                    match &mut result {
                        Some(existing) => merge_repr(existing, current),
                        None => result = Some(current),
                    }
                }
                ControlFlow::Continue(())
            });
            result
        }
    }

    /// Call this only if there are legacy const generics, to save memory.
    #[salsa::tracked(returns(ref))]
    pub(crate) fn legacy_const_generic_indices(
        db: &dyn DefDatabase,
        owner: FunctionId,
    ) -> Option<Box<[u32]>> {
        let result = collect_attrs(db, owner.into(), |attr| {
            if let ast::Meta::TokenTreeMeta(attr) = attr
                && attr.path().is1("rustc_legacy_const_generics")
                && let Some(tt) = attr.token_tree()
            {
                let result = parse_rustc_legacy_const_generics(tt);
                ControlFlow::Break(result)
            } else {
                ControlFlow::Continue(())
            }
        });
        result.filter(|it| !it.is_empty())
    }

    // There aren't typically many crates, so it's okay to always make this a query without a flag.
    #[salsa::tracked(returns(ref))]
    pub fn doc_html_root_url(db: &dyn DefDatabase, krate: Crate) -> Option<SmolStr> {
        let root_file_id = krate.root_file_id(db);
        let syntax = root_file_id.parse(db).tree();
        let extra_crate_attrs =
            parse_extra_crate_attrs(db, krate).into_iter().flat_map(|src| src.attrs());

        let mut cfg_options = None;
        expand_cfg_attr(
            extra_crate_attrs.chain(syntax.attrs()),
            || cfg_options.get_or_insert(krate.cfg_options(db)),
            |attr, _| {
                if let ast::Meta::TokenTreeMeta(attr) = attr
                    && attr.path().is1("doc")
                    && let Some(tt) = attr.token_tree()
                    && let Some(result) = DocAtom::parse(tt).into_iter().find_map(|atom| {
                        if let DocAtom::KeyValue { key, value } = atom
                            && key == "html_root_url"
                        {
                            Some(value)
                        } else {
                            None
                        }
                    })
                {
                    ControlFlow::Break(result)
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
    }

    #[inline]
    pub fn target_features(db: &dyn DefDatabase, owner: FunctionId) -> &FxHashSet<Symbol> {
        if !AttrFlags::query(db, owner.into()).contains(AttrFlags::HAS_TARGET_FEATURE) {
            return const { &FxHashSet::with_hasher(rustc_hash::FxBuildHasher) };
        }

        return target_features(db, owner);

        #[salsa::tracked(returns(ref))]
        fn target_features(db: &dyn DefDatabase, owner: FunctionId) -> FxHashSet<Symbol> {
            let mut result = FxHashSet::default();
            collect_attrs::<Infallible>(db, owner.into(), |attr| {
                if let ast::Meta::TokenTreeMeta(attr) = attr
                    && attr.path().is1("target_feature")
                    && let Some(tt) = attr.token_tree()
                {
                    let mut tt = TokenTreeChildren::new(&tt);
                    while let Some(NodeOrToken::Token(enable_ident)) = tt.next()
                        && enable_ident.text() == "enable"
                        && let Some(NodeOrToken::Token(eq_token)) = tt.next()
                        && eq_token.kind() == T![=]
                        && let Some(NodeOrToken::Token(features)) = tt.next()
                        && let Some(features) = ast::String::cast(features)
                        && let Ok(features) = features.value()
                    {
                        result.extend(features.split(',').map(Symbol::intern));
                        if tt
                            .next()
                            .and_then(NodeOrToken::into_token)
                            .is_none_or(|it| it.kind() != T![,])
                        {
                            break;
                        }
                    }
                }
                ControlFlow::Continue(())
            });
            result.shrink_to_fit();
            result
        }
    }

    #[inline]
    pub fn rustc_layout_scalar_valid_range(
        db: &dyn DefDatabase,
        owner: AdtId,
    ) -> RustcLayoutScalarValidRange {
        if !AttrFlags::query(db, owner.into()).contains(AttrFlags::RUSTC_LAYOUT_SCALAR_VALID_RANGE)
        {
            return RustcLayoutScalarValidRange::default();
        }

        return rustc_layout_scalar_valid_range(db, owner);

        #[salsa::tracked]
        fn rustc_layout_scalar_valid_range(
            db: &dyn DefDatabase,
            owner: AdtId,
        ) -> RustcLayoutScalarValidRange {
            let mut result = RustcLayoutScalarValidRange::default();
            collect_attrs::<Infallible>(db, owner.into(), |attr| {
                if let ast::Meta::TokenTreeMeta(attr) = attr
                    && let path = attr.path()
                    && (path.is1("rustc_layout_scalar_valid_range_start")
                        || path.is1("rustc_layout_scalar_valid_range_end"))
                    && let Some(tt) = attr.token_tree()
                    && let tt = TokenTreeChildren::new(&tt)
                    && let Ok(NodeOrToken::Token(value)) = Itertools::exactly_one(tt)
                    && let Some(value) = ast::IntNumber::cast(value)
                    && let Ok(value) = value.value()
                {
                    if path.is1("rustc_layout_scalar_valid_range_start") {
                        result.start = Some(value)
                    } else {
                        result.end = Some(value);
                    }
                }
                ControlFlow::Continue(())
            });
            result
        }
    }

    #[inline]
    pub fn doc_aliases(self, db: &dyn DefDatabase, owner: Either<AttrDefId, FieldId>) -> &[Symbol] {
        if !self.contains(AttrFlags::HAS_DOC_ALIASES) {
            return &[];
        }
        return match owner {
            Either::Left(it) => doc_aliases(db, it),
            Either::Right(field) => fields_doc_aliases(db, field.parent)
                .get(field.local_id)
                .map(|it| &**it)
                .unwrap_or_default(),
        };

        #[salsa::tracked(returns(ref))]
        fn doc_aliases(db: &dyn DefDatabase, owner: AttrDefId) -> Box<[Symbol]> {
            let mut result = Vec::new();
            collect_attrs::<Infallible>(db, owner, |attr| extract_doc_aliases(&mut result, attr));
            result.into_boxed_slice()
        }

        #[salsa::tracked(returns(ref))]
        fn fields_doc_aliases(
            db: &dyn DefDatabase,
            variant: VariantId,
        ) -> ArenaMap<LocalFieldId, Box<[Symbol]>> {
            collect_field_attrs(db, variant, |cfg_options, field| {
                let mut result = Vec::new();
                expand_cfg_attr(
                    field.value.attrs(),
                    || cfg_options,
                    |attr, _| extract_doc_aliases(&mut result, attr),
                );
                result.into_boxed_slice()
            })
        }
    }

    #[inline]
    pub fn cfgs(self, db: &dyn DefDatabase, owner: Either<AttrDefId, FieldId>) -> Option<&CfgExpr> {
        if !self.contains(AttrFlags::HAS_CFG) {
            return None;
        }
        return match owner {
            Either::Left(it) => cfgs(db, it).as_ref(),
            Either::Right(field) => {
                fields_cfgs(db, field.parent).get(field.local_id).and_then(|it| it.as_ref())
            }
        };

        // We LRU this query because it is only used by IDE.
        #[salsa::tracked(returns(ref), lru = 250)]
        fn cfgs(db: &dyn DefDatabase, owner: AttrDefId) -> Option<CfgExpr> {
            let mut result = Vec::new();
            collect_attrs::<Infallible>(db, owner, |attr| extract_cfgs(&mut result, attr));
            match result.len() {
                0 => None,
                1 => result.into_iter().next(),
                _ => Some(CfgExpr::All(result.into_boxed_slice())),
            }
        }

        // We LRU this query because it is only used by IDE.
        #[salsa::tracked(returns(ref), lru = 50)]
        fn fields_cfgs(
            db: &dyn DefDatabase,
            variant: VariantId,
        ) -> ArenaMap<LocalFieldId, Option<CfgExpr>> {
            collect_field_attrs(db, variant, |cfg_options, field| {
                let mut result = Vec::new();
                expand_cfg_attr(
                    field.value.attrs(),
                    || cfg_options,
                    |attr, _| extract_cfgs(&mut result, attr),
                );
                match result.len() {
                    0 => None,
                    1 => result.into_iter().next(),
                    _ => Some(CfgExpr::All(result.into_boxed_slice())),
                }
            })
        }
    }

    #[inline]
    pub fn doc_keyword(db: &dyn DefDatabase, owner: ModuleId) -> Option<Symbol> {
        if !AttrFlags::query(db, AttrDefId::ModuleId(owner)).contains(AttrFlags::HAS_DOC_KEYWORD) {
            return None;
        }
        return doc_keyword(db, owner);

        #[salsa::tracked]
        fn doc_keyword(db: &dyn DefDatabase, owner: ModuleId) -> Option<Symbol> {
            collect_attrs(db, AttrDefId::ModuleId(owner), |attr| {
                if let ast::Meta::TokenTreeMeta(attr) = attr
                    && attr.path().is1("doc")
                    && let Some(tt) = attr.token_tree()
                {
                    for atom in DocAtom::parse(tt) {
                        if let DocAtom::KeyValue { key, value } = atom
                            && key == "keyword"
                        {
                            return ControlFlow::Break(Symbol::intern(&value));
                        }
                    }
                }
                ControlFlow::Continue(())
            })
        }
    }

    // We LRU this query because it is only used by IDE.
    #[salsa::tracked(returns(ref), lru = 250)]
    pub fn docs(db: &dyn DefDatabase, owner: AttrDefId) -> Option<Box<Docs>> {
        let (source, outer_mod_decl, _extra_crate_attrs, krate) = attrs_source(db, owner);
        let inner_attrs_node = source.value.inner_attributes_node();
        // Note: we don't have to pass down `_extra_crate_attrs` here, since `extract_docs`
        // does not handle crate-level attributes related to docs.
        // See: https://doc.rust-lang.org/rustdoc/write-documentation/the-doc-attribute.html#at-the-crate-level
        self::docs::extract_docs(
            db,
            krate,
            &|| resolver_for_attr_def_id(db, owner),
            &|| krate.cfg_options(db),
            source,
            outer_mod_decl,
            inner_attrs_node,
        )
    }

    #[inline]
    pub fn field_docs(db: &dyn DefDatabase, field: FieldId) -> Option<&Docs> {
        return fields_docs(db, field.parent).get(field.local_id).and_then(|it| it.as_deref());

        // We LRU this query because it is only used by IDE.
        #[salsa::tracked(returns(ref), lru = 50)]
        pub fn fields_docs(
            db: &dyn DefDatabase,
            variant: VariantId,
        ) -> ArenaMap<LocalFieldId, Option<Box<Docs>>> {
            let krate = variant.module(db).krate(db);
            collect_field_attrs(db, variant, |cfg_options, field| {
                self::docs::extract_docs(
                    db,
                    krate,
                    &|| variant.resolver(db),
                    &|| cfg_options,
                    field,
                    None,
                    None,
                )
            })
        }
    }

    #[inline]
    pub fn derive_info(db: &dyn DefDatabase, owner: MacroId) -> Option<&DeriveInfo> {
        if !AttrFlags::query(db, owner.into()).contains(AttrFlags::IS_DERIVE_OR_BUILTIN_MACRO) {
            return None;
        }

        return derive_info(db, owner).as_ref();

        #[salsa::tracked(returns(ref))]
        fn derive_info(db: &dyn DefDatabase, owner: MacroId) -> Option<DeriveInfo> {
            collect_attrs(db, owner.into(), |attr| {
                if let ast::Meta::TokenTreeMeta(attr) = attr
                    && (attr.path().is1("proc_macro_derive")
                        || attr.path().is1("rustc_builtin_macro"))
                    && let Some(tt) = attr.token_tree()
                    && let mut tt = TokenTreeChildren::new(&tt)
                    && let Some(NodeOrToken::Token(trait_name)) = tt.next()
                    && trait_name.kind().is_any_identifier()
                {
                    let trait_name = Symbol::intern(trait_name.text());

                    let helpers = if let Some(NodeOrToken::Token(comma)) = tt.next()
                        && comma.kind() == T![,]
                        && let Some(NodeOrToken::Token(attributes)) = tt.next()
                        && attributes.text() == "attributes"
                        && let Some(NodeOrToken::Node(attributes)) = tt.next()
                    {
                        attributes
                            .syntax()
                            .children_with_tokens()
                            .filter_map(NodeOrToken::into_token)
                            .filter(|it| it.kind().is_any_identifier())
                            .map(|it| Symbol::intern(it.text()))
                            .collect::<Box<[_]>>()
                    } else {
                        Box::new([])
                    };

                    ControlFlow::Break(DeriveInfo { trait_name, helpers })
                } else {
                    ControlFlow::Continue(())
                }
            })
        }
    }

    pub fn unstable_feature(self, db: &dyn DefDatabase, owner: AttrDefId) -> Option<Symbol> {
        if !self.contains(AttrFlags::IS_UNSTABLE) {
            return None;
        }

        return unstable_feature(db, owner);

        #[salsa::tracked]
        fn unstable_feature(db: &dyn DefDatabase, owner: AttrDefId) -> Option<Symbol> {
            collect_attrs(db, owner, |attr| {
                if let ast::Meta::TokenTreeMeta(attr) = attr
                    && let path = attr.path()
                    && path.is1("unstable")
                    && let Some(tt) = attr.token_tree()
                {
                    let mut tt = TokenTreeChildren::new(&tt);
                    // Technically the `feature = "..."` always comes first, but it's not a requirement.
                    while let Some(token) = tt.next() {
                        if let NodeOrToken::Token(token) = token
                            && token.text() == "feature"
                            && let Some(NodeOrToken::Token(eq)) = tt.next()
                            && eq.kind() == T![=]
                            && let Some(NodeOrToken::Token(feature)) = tt.next()
                            && let Some(feature) = ast::String::cast(feature)
                            && let Ok(feature) = feature.value()
                        {
                            return ControlFlow::Break(Symbol::intern(&feature));
                        }
                    }
                }
                ControlFlow::Continue(())
            })
        }
    }
}

fn merge_repr(this: &mut ReprOptions, other: ReprOptions) {
    let ReprOptions { int, align, pack, flags, scalable, field_shuffle_seed: _ } = this;
    flags.insert(other.flags);
    *align = (*align).max(other.align);
    *pack = match (*pack, other.pack) {
        (Some(pack), None) | (None, Some(pack)) => Some(pack),
        _ => (*pack).min(other.pack),
    };
    if other.int.is_some() {
        *int = other.int;
    }
    if other.scalable.is_some() {
        *scalable = other.scalable;
    }
}

fn parse_repr_tt(tt: &ast::TokenTree) -> Option<ReprOptions> {
    use crate::builtin_type::{BuiltinInt, BuiltinUint};
    use rustc_abi::{Align, Integer, IntegerType, ReprFlags, ReprOptions};

    let mut tts = TokenTreeChildren::new(tt).peekable();

    let mut acc = ReprOptions::default();
    while let Some(tt) = tts.next() {
        let NodeOrToken::Token(ident) = tt else {
            continue;
        };
        if !ident.kind().is_any_identifier() {
            continue;
        }
        let repr = match ident.text() {
            "packed" => {
                let pack = if let Some(NodeOrToken::Node(tt)) = tts.peek() {
                    let tt = tt.clone();
                    tts.next();
                    let mut tt_iter = TokenTreeChildren::new(&tt);
                    if let Some(NodeOrToken::Token(lit)) = tt_iter.next()
                        && let Some(lit) = ast::IntNumber::cast(lit)
                        && let Ok(lit) = lit.value()
                        && let Ok(lit) = lit.try_into()
                    {
                        lit
                    } else {
                        0
                    }
                } else {
                    0
                };
                let pack = Some(Align::from_bytes(pack).unwrap_or(Align::ONE));
                ReprOptions { pack, ..Default::default() }
            }
            "align" => {
                let mut align = None;
                if let Some(NodeOrToken::Node(tt)) = tts.peek() {
                    let tt = tt.clone();
                    tts.next();
                    let mut tt_iter = TokenTreeChildren::new(&tt);
                    if let Some(NodeOrToken::Token(lit)) = tt_iter.next()
                        && let Some(lit) = ast::IntNumber::cast(lit)
                        && let Ok(lit) = lit.value()
                        && let Ok(lit) = lit.try_into()
                    {
                        align = Align::from_bytes(lit).ok();
                    }
                }
                ReprOptions { align, ..Default::default() }
            }
            "C" => ReprOptions { flags: ReprFlags::IS_C, ..Default::default() },
            "transparent" => ReprOptions { flags: ReprFlags::IS_TRANSPARENT, ..Default::default() },
            "simd" => ReprOptions { flags: ReprFlags::IS_SIMD, ..Default::default() },
            repr => {
                let mut int = None;
                if let Some(builtin) = BuiltinInt::from_suffix(repr)
                    .map(Either::Left)
                    .or_else(|| BuiltinUint::from_suffix(repr).map(Either::Right))
                {
                    int = Some(match builtin {
                        Either::Left(bi) => match bi {
                            BuiltinInt::Isize => IntegerType::Pointer(true),
                            BuiltinInt::I8 => IntegerType::Fixed(Integer::I8, true),
                            BuiltinInt::I16 => IntegerType::Fixed(Integer::I16, true),
                            BuiltinInt::I32 => IntegerType::Fixed(Integer::I32, true),
                            BuiltinInt::I64 => IntegerType::Fixed(Integer::I64, true),
                            BuiltinInt::I128 => IntegerType::Fixed(Integer::I128, true),
                        },
                        Either::Right(bu) => match bu {
                            BuiltinUint::Usize => IntegerType::Pointer(false),
                            BuiltinUint::U8 => IntegerType::Fixed(Integer::I8, false),
                            BuiltinUint::U16 => IntegerType::Fixed(Integer::I16, false),
                            BuiltinUint::U32 => IntegerType::Fixed(Integer::I32, false),
                            BuiltinUint::U64 => IntegerType::Fixed(Integer::I64, false),
                            BuiltinUint::U128 => IntegerType::Fixed(Integer::I128, false),
                        },
                    });
                }
                ReprOptions { int, ..Default::default() }
            }
        };
        merge_repr(&mut acc, repr);
    }

    Some(acc)
}

fn parse_rustc_legacy_const_generics(tt: ast::TokenTree) -> Box<[u32]> {
    TokenTreeChildren::new(&tt)
        .filter_map(|param| {
            ast::IntNumber::cast(param.into_token()?)?.value().ok()?.try_into().ok()
        })
        .collect()
}

#[derive(Debug)]
enum DocAtom {
    /// eg. `#[doc(hidden)]`
    Flag(SmolStr),
    /// eg. `#[doc(alias = "it")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[doc(alias = "x")]` and `#[doc(alias = "y")]`.
    KeyValue { key: SmolStr, value: SmolStr },
    /// eg. `#[doc(alias("x", "y"))]`
    Alias(Vec<SmolStr>),
}

impl DocAtom {
    fn parse(tt: ast::TokenTree) -> SmallVec<[DocAtom; 1]> {
        let mut iter = TokenTreeChildren::new(&tt).peekable();
        let mut result = SmallVec::new();
        while iter.peek().is_some() {
            if let Some(expr) = next_doc_expr(&mut iter) {
                result.push(expr);
            }
        }
        result
    }
}

fn next_doc_expr(it: &mut Peekable<TokenTreeChildren>) -> Option<DocAtom> {
    let name = match it.next() {
        Some(NodeOrToken::Token(token)) if token.kind().is_any_identifier() => {
            SmolStr::new(token.text())
        }
        _ => return None,
    };

    let ret = match it.peek() {
        Some(NodeOrToken::Token(eq)) if eq.kind() == T![=] => {
            it.next();
            if let Some(NodeOrToken::Token(value)) = it.next()
                && let Some(value) = ast::String::cast(value)
                && let Ok(value) = value.value()
            {
                DocAtom::KeyValue { key: name, value: SmolStr::new(&*value) }
            } else {
                return None;
            }
        }
        Some(NodeOrToken::Node(subtree)) => {
            if name != "alias" {
                return None;
            }
            let aliases = TokenTreeChildren::new(subtree)
                .filter_map(|alias| {
                    Some(SmolStr::new(&*ast::String::cast(alias.into_token()?)?.value().ok()?))
                })
                .collect();
            it.next();
            DocAtom::Alias(aliases)
        }
        _ => DocAtom::Flag(name),
    };
    Some(ret)
}

#[cfg(test)]
mod tests {
    use test_fixture::WithFixture;

    use crate::AttrDefId;
    use crate::attrs::AttrFlags;
    use crate::test_db::TestDB;

    #[test]
    fn crate_attrs() {
        let fixture = r#"
//- /lib.rs crate:foo crate-attr:no_std crate-attr:cfg(target_arch="x86")
        "#;
        let (db, file_id) = TestDB::with_single_file(fixture);
        let module = db.module_for_file(file_id.file_id(&db));
        let attrs = AttrFlags::query(&db, AttrDefId::ModuleId(module));
        assert!(attrs.contains(AttrFlags::IS_NO_STD | AttrFlags::HAS_CFG));
    }
}
