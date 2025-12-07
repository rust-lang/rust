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
//! An exception to this model that is also defined in this module is documentation (doc
//! comments and `#[doc = "..."]` attributes). But it also has a more compact form than
//! the attribute: a concatenated string of the full docs as well as a source map
//! to map it back to AST (which is needed for things like resolving links in doc comments
//! and highlight injection). The lowering and upmapping of doc comments is a bit complicated,
//! but it is encapsulated in the [`Docs`] struct.

use std::{
    convert::Infallible,
    iter::Peekable,
    ops::{ControlFlow, Range},
};

use base_db::Crate;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    HirFileId, InFile, Lookup,
    attrs::{Meta, expand_cfg_attr, expand_cfg_attr_with_doc_comments},
};
use intern::Symbol;
use itertools::Itertools;
use la_arena::ArenaMap;
use rustc_abi::ReprOptions;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use syntax::{
    AstNode, AstToken, NodeOrToken, SmolStr, SyntaxNode, SyntaxToken, T,
    ast::{self, AttrDocCommentIter, HasAttrs, IsString, TokenTreeChildren},
};
use tt::{TextRange, TextSize};

use crate::{
    AdtId, AstIdLoc, AttrDefId, FieldId, FunctionId, GenericDefId, HasModule, LifetimeParamId,
    LocalFieldId, MacroId, ModuleId, TypeOrConstParamId, VariantId,
    db::DefDatabase,
    hir::generics::{GenericParams, LocalLifetimeParamId, LocalTypeOrConstParamId},
    nameres::ModuleOrigin,
    src::{HasChildSource, HasSource},
};

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
fn match_attr_flags(attr_flags: &mut AttrFlags, attr: Meta) -> ControlFlow<Infallible> {
    match attr {
        Meta::NamedKeyValue { name: Some(name), value, .. } => match name.text() {
            "deprecated" => attr_flags.insert(AttrFlags::IS_DEPRECATED),
            "lang" => attr_flags.insert(AttrFlags::LANG_ITEM),
            "path" => attr_flags.insert(AttrFlags::HAS_PATH),
            "unstable" => attr_flags.insert(AttrFlags::IS_UNSTABLE),
            "export_name" => {
                if let Some(value) = value
                    && let Some(value) = ast::String::cast(value)
                    && let Ok(value) = value.value()
                    && *value == *"main"
                {
                    attr_flags.insert(AttrFlags::IS_EXPORT_NAME_MAIN);
                }
            }
            _ => {}
        },
        Meta::TokenTree { path, tt } => match path.segments.len() {
            1 => match path.segments[0].text() {
                "deprecated" => attr_flags.insert(AttrFlags::IS_DEPRECATED),
                "cfg" => attr_flags.insert(AttrFlags::HAS_CFG),
                "doc" => extract_doc_tt_attr(attr_flags, tt),
                "repr" => attr_flags.insert(AttrFlags::HAS_REPR),
                "target_feature" => attr_flags.insert(AttrFlags::HAS_TARGET_FEATURE),
                "proc_macro_derive" | "rustc_builtin_macro" => {
                    attr_flags.insert(AttrFlags::IS_DERIVE_OR_BUILTIN_MACRO)
                }
                "unstable" => attr_flags.insert(AttrFlags::IS_UNSTABLE),
                "rustc_layout_scalar_valid_range_start" | "rustc_layout_scalar_valid_range_end" => {
                    attr_flags.insert(AttrFlags::RUSTC_LAYOUT_SCALAR_VALID_RANGE)
                }
                "rustc_legacy_const_generics" => {
                    attr_flags.insert(AttrFlags::HAS_LEGACY_CONST_GENERICS)
                }
                "rustc_skip_during_method_dispatch" => {
                    extract_rustc_skip_during_method_dispatch(attr_flags, tt)
                }
                _ => {}
            },
            2 => match path.segments[0].text() {
                "rust_analyzer" => match path.segments[1].text() {
                    "completions" => extract_ra_completions(attr_flags, tt),
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        },
        Meta::Path { path } => {
            match path.segments.len() {
                1 => match path.segments[0].text() {
                    "rustc_has_incoherent_inherent_impls" => {
                        attr_flags.insert(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS)
                    }
                    "rustc_allow_incoherent_impl" => {
                        attr_flags.insert(AttrFlags::RUSTC_ALLOW_INCOHERENT_IMPL)
                    }
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
                2 => match path.segments[0].text() {
                    "rust_analyzer" => match path.segments[1].text() {
                        "skip" => attr_flags.insert(AttrFlags::RUST_ANALYZER_SKIP),
                        _ => {}
                    },
                    _ => {}
                },
                _ => {}
            }

            if path.is_test {
                attr_flags.insert(AttrFlags::IS_TEST);
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
    }
}

fn attrs_source(
    db: &dyn DefDatabase,
    owner: AttrDefId,
) -> (InFile<ast::AnyHasAttrs>, Option<InFile<ast::Module>>, Crate) {
    let (owner, krate) = match owner {
        AttrDefId::ModuleId(id) => {
            let def_map = id.def_map(db);
            let (definition, declaration) = match def_map[id].origin {
                ModuleOrigin::CrateRoot { definition } => {
                    let file = db.parse(definition).tree();
                    (InFile::new(definition.into(), ast::AnyHasAttrs::from(file)), None)
                }
                ModuleOrigin::File { declaration, declaration_tree_id, definition, .. } => {
                    let declaration = InFile::new(declaration_tree_id.file_id(), declaration);
                    let declaration = declaration.with_value(declaration.to_node(db));
                    let definition_source = db.parse(definition).tree();
                    (InFile::new(definition.into(), definition_source.into()), Some(declaration))
                }
                ModuleOrigin::Inline { definition_tree_id, definition } => {
                    let definition = InFile::new(definition_tree_id.file_id(), definition);
                    let definition = definition.with_value(definition.to_node(db).into());
                    (definition, None)
                }
                ModuleOrigin::BlockExpr { block, .. } => {
                    let definition = block.to_node(db);
                    (block.with_value(definition.into()), None)
                }
            };
            return (definition, declaration, def_map.krate());
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
    (owner, None, krate)
}

fn collect_attrs<BreakValue>(
    db: &dyn DefDatabase,
    owner: AttrDefId,
    mut callback: impl FnMut(Meta) -> ControlFlow<BreakValue>,
) -> Option<BreakValue> {
    let (source, outer_mod_decl, krate) = attrs_source(db, owner);

    let mut cfg_options = None;
    expand_cfg_attr(
        outer_mod_decl
            .into_iter()
            .flat_map(|it| it.value.attrs())
            .chain(ast::attrs_including_inner(&source.value)),
        || cfg_options.get_or_insert_with(|| krate.cfg_options(db)),
        move |meta, _, _, _| callback(meta),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DocsSourceMapLine {
    /// The offset in [`Docs::docs`].
    string_offset: TextSize,
    /// The offset in the AST of the text.
    ast_offset: TextSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Docs {
    /// The concatenated string of all `#[doc = "..."]` attributes and documentation comments.
    docs: String,
    /// A sorted map from an offset in `docs` to an offset in the source code.
    docs_source_map: Vec<DocsSourceMapLine>,
    /// If the item is an outlined module (`mod foo;`), `docs_source_map` store the concatenated
    /// list of the outline and inline docs (outline first). Then, this field contains the [`HirFileId`]
    /// of the outline declaration, and the index in `docs` from which the inline docs
    /// begin.
    outline_mod: Option<(HirFileId, usize)>,
    inline_file: HirFileId,
    /// The size the prepended prefix, which does not map to real doc comments.
    prefix_len: TextSize,
    /// The offset in `docs` from which the docs are inner attributes/comments.
    inline_inner_docs_start: Option<TextSize>,
    /// Like `inline_inner_docs_start`, but for `outline_mod`. This can happen only when merging `Docs`
    /// (as outline modules don't have inner attributes).
    outline_inner_docs_start: Option<TextSize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsInnerDoc {
    No,
    Yes,
}

impl IsInnerDoc {
    #[inline]
    pub fn yes(self) -> bool {
        self == IsInnerDoc::Yes
    }
}

impl Docs {
    #[inline]
    pub fn docs(&self) -> &str {
        &self.docs
    }

    #[inline]
    pub fn into_docs(self) -> String {
        self.docs
    }

    pub fn find_ast_range(
        &self,
        mut string_range: TextRange,
    ) -> Option<(InFile<TextRange>, IsInnerDoc)> {
        if string_range.start() < self.prefix_len {
            return None;
        }
        string_range -= self.prefix_len;

        let mut file = self.inline_file;
        let mut inner_docs_start = self.inline_inner_docs_start;
        // Check whether the range is from the outline, the inline, or both.
        let source_map = if let Some((outline_mod_file, outline_mod_end)) = self.outline_mod {
            if let Some(first_inline) = self.docs_source_map.get(outline_mod_end) {
                if string_range.end() <= first_inline.string_offset {
                    // The range is completely in the outline.
                    file = outline_mod_file;
                    inner_docs_start = self.outline_inner_docs_start;
                    &self.docs_source_map[..outline_mod_end]
                } else if string_range.start() >= first_inline.string_offset {
                    // The range is completely in the inline.
                    &self.docs_source_map[outline_mod_end..]
                } else {
                    // The range is combined from the outline and the inline - cannot map it back.
                    return None;
                }
            } else {
                // There is no inline.
                file = outline_mod_file;
                inner_docs_start = self.outline_inner_docs_start;
                &self.docs_source_map
            }
        } else {
            // There is no outline.
            &self.docs_source_map
        };

        let after_range =
            source_map.partition_point(|line| line.string_offset <= string_range.start()) - 1;
        let after_range = &source_map[after_range..];
        let line = after_range.first()?;
        if after_range.get(1).is_some_and(|next_line| next_line.string_offset < string_range.end())
        {
            // The range is combined from two lines - cannot map it back.
            return None;
        }
        let ast_range = string_range - line.string_offset + line.ast_offset;
        let is_inner = if inner_docs_start
            .is_some_and(|inner_docs_start| string_range.start() >= inner_docs_start)
        {
            IsInnerDoc::Yes
        } else {
            IsInnerDoc::No
        };
        Some((InFile::new(file, ast_range), is_inner))
    }

    #[inline]
    pub fn shift_by(&mut self, offset: TextSize) {
        self.prefix_len += offset;
    }

    pub fn prepend_str(&mut self, s: &str) {
        self.prefix_len += TextSize::of(s);
        self.docs.insert_str(0, s);
    }

    pub fn append_str(&mut self, s: &str) {
        self.docs.push_str(s);
    }

    pub fn append(&mut self, other: &Docs) {
        let other_offset = TextSize::of(&self.docs);

        assert!(
            self.outline_mod.is_none() && other.outline_mod.is_none(),
            "cannot merge `Docs` that have `outline_mod` set"
        );
        self.outline_mod = Some((self.inline_file, self.docs_source_map.len()));
        self.inline_file = other.inline_file;
        self.outline_inner_docs_start = self.inline_inner_docs_start;
        self.inline_inner_docs_start = other.inline_inner_docs_start.map(|it| it + other_offset);

        self.docs.push_str(&other.docs);
        self.docs_source_map.extend(other.docs_source_map.iter().map(
            |&DocsSourceMapLine { string_offset, ast_offset }| DocsSourceMapLine {
                ast_offset,
                string_offset: string_offset + other_offset,
            },
        ));
    }

    fn extend_with_doc_comment(&mut self, comment: ast::Comment, indent: &mut usize) {
        let Some((doc, offset)) = comment.doc_comment() else { return };
        self.extend_with_doc_str(doc, comment.syntax().text_range().start() + offset, indent);
    }

    fn extend_with_doc_attr(&mut self, value: SyntaxToken, indent: &mut usize) {
        let Some(value) = ast::String::cast(value) else { return };
        let Some(value_offset) = value.text_range_between_quotes() else { return };
        let value_offset = value_offset.start();
        let Ok(value) = value.value() else { return };
        // FIXME: Handle source maps for escaped text.
        self.extend_with_doc_str(&value, value_offset, indent);
    }

    fn extend_with_doc_str(&mut self, doc: &str, mut offset_in_ast: TextSize, indent: &mut usize) {
        for line in doc.split('\n') {
            self.docs_source_map.push(DocsSourceMapLine {
                string_offset: TextSize::of(&self.docs),
                ast_offset: offset_in_ast,
            });
            offset_in_ast += TextSize::of(line) + TextSize::of("\n");

            let line = line.trim_end();
            if let Some(line_indent) = line.chars().position(|ch| !ch.is_whitespace()) {
                // Empty lines are handled because `position()` returns `None` for them.
                *indent = std::cmp::min(*indent, line_indent);
            }
            self.docs.push_str(line);
            self.docs.push('\n');
        }
    }

    fn remove_indent(&mut self, indent: usize, start_source_map_index: usize) {
        /// In case of panics, we want to avoid corrupted UTF-8 in `self.docs`, so we clear it.
        struct Guard<'a>(&'a mut Docs);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                let Docs {
                    docs,
                    docs_source_map,
                    outline_mod,
                    inline_file: _,
                    prefix_len: _,
                    inline_inner_docs_start: _,
                    outline_inner_docs_start: _,
                } = self.0;
                // Don't use `String::clear()` here because it's not guaranteed to not do UTF-8-dependent things,
                // and we may have temporarily broken the string's encoding.
                unsafe { docs.as_mut_vec() }.clear();
                // This is just to avoid panics down the road.
                docs_source_map.clear();
                *outline_mod = None;
            }
        }

        if self.docs.is_empty() {
            return;
        }

        let guard = Guard(self);
        let source_map = &mut guard.0.docs_source_map[start_source_map_index..];
        let Some(&DocsSourceMapLine { string_offset: mut copy_into, .. }) = source_map.first()
        else {
            return;
        };
        // We basically want to remove multiple ranges from a string. Doing this efficiently (without O(N^2)
        // or allocations) requires unsafe. Basically, for each line, we copy the line minus the indent into
        // consecutive to the previous line (which may have moved). Then at the end we truncate.
        let mut accumulated_offset = TextSize::new(0);
        for idx in 0..source_map.len() {
            let string_end_offset = source_map
                .get(idx + 1)
                .map_or_else(|| TextSize::of(&guard.0.docs), |next_attr| next_attr.string_offset);
            let line_source = &mut source_map[idx];
            let line_docs =
                &guard.0.docs[TextRange::new(line_source.string_offset, string_end_offset)];
            let line_docs_len = TextSize::of(line_docs);
            let indent_size = line_docs.char_indices().nth(indent).map_or_else(
                || TextSize::of(line_docs) - TextSize::of("\n"),
                |(offset, _)| TextSize::new(offset as u32),
            );
            unsafe { guard.0.docs.as_bytes_mut() }.copy_within(
                Range::<usize>::from(TextRange::new(
                    line_source.string_offset + indent_size,
                    string_end_offset,
                )),
                copy_into.into(),
            );
            copy_into += line_docs_len - indent_size;

            if let Some(inner_attrs_start) = &mut guard.0.inline_inner_docs_start
                && *inner_attrs_start == line_source.string_offset
            {
                *inner_attrs_start -= accumulated_offset;
            }
            // The removals in the string accumulate, but in the AST not, because it already points
            // to the beginning of each attribute.
            // Also, we need to shift the AST offset of every line, but the string offset of the first
            // line should not get shifted (in general, the shift for the string offset is by the
            // number of lines until the current one, excluding the current one).
            line_source.string_offset -= accumulated_offset;
            line_source.ast_offset += indent_size;

            accumulated_offset += indent_size;
        }
        // Don't use `String::truncate()` here because it's not guaranteed to not do UTF-8-dependent things,
        // and we may have temporarily broken the string's encoding.
        unsafe { guard.0.docs.as_mut_vec() }.truncate(copy_into.into());

        std::mem::forget(guard);
    }

    fn remove_last_newline(&mut self) {
        self.docs.truncate(self.docs.len().saturating_sub(1));
    }

    fn shrink_to_fit(&mut self) {
        let Docs {
            docs,
            docs_source_map,
            outline_mod: _,
            inline_file: _,
            prefix_len: _,
            inline_inner_docs_start: _,
            outline_inner_docs_start: _,
        } = self;
        docs.shrink_to_fit();
        docs_source_map.shrink_to_fit();
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DeriveInfo {
    pub trait_name: Symbol,
    pub helpers: Box<[Symbol]>,
}

fn extract_doc_aliases(result: &mut Vec<Symbol>, attr: Meta) -> ControlFlow<Infallible> {
    if let Meta::TokenTree { path, tt } = attr
        && path.is1("doc")
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

fn extract_cfgs(result: &mut Vec<CfgExpr>, attr: Meta) -> ControlFlow<Infallible> {
    if let Meta::TokenTree { path, tt } = attr
        && path.is1("cfg")
    {
        result.push(CfgExpr::parse_from_ast(&mut TokenTreeChildren::new(&tt).peekable()));
    }
    ControlFlow::Continue(())
}

fn extract_docs<'a>(
    get_cfg_options: &dyn Fn() -> &'a CfgOptions,
    source: InFile<ast::AnyHasAttrs>,
    outer_mod_decl: Option<InFile<ast::Module>>,
    inner_attrs_node: Option<SyntaxNode>,
) -> Option<Box<Docs>> {
    let mut result = Docs {
        docs: String::new(),
        docs_source_map: Vec::new(),
        outline_mod: None,
        inline_file: source.file_id,
        prefix_len: TextSize::new(0),
        inline_inner_docs_start: None,
        outline_inner_docs_start: None,
    };

    let mut cfg_options = None;
    let mut extend_with_attrs =
        |result: &mut Docs, node: &SyntaxNode, expect_inner_attrs, indent: &mut usize| {
            expand_cfg_attr_with_doc_comments::<_, Infallible>(
                AttrDocCommentIter::from_syntax_node(node).filter(|attr| match attr {
                    Either::Left(attr) => attr.kind().is_inner() == expect_inner_attrs,
                    Either::Right(comment) => comment.kind().doc.is_some_and(|kind| {
                        (kind == ast::CommentPlacement::Inner) == expect_inner_attrs
                    }),
                }),
                || cfg_options.get_or_insert_with(get_cfg_options),
                |attr| {
                    match attr {
                        Either::Right(doc_comment) => {
                            result.extend_with_doc_comment(doc_comment, indent)
                        }
                        Either::Left((attr, _, _, _)) => match attr {
                            // FIXME: Handle macros: `#[doc = concat!("foo", "bar")]`.
                            Meta::NamedKeyValue {
                                name: Some(name), value: Some(value), ..
                            } if name.text() == "doc" => {
                                result.extend_with_doc_attr(value, indent);
                            }
                            _ => {}
                        },
                    }
                    ControlFlow::Continue(())
                },
            );
        };

    if let Some(outer_mod_decl) = outer_mod_decl {
        let mut indent = usize::MAX;
        extend_with_attrs(&mut result, outer_mod_decl.value.syntax(), false, &mut indent);
        result.remove_indent(indent, 0);
        result.outline_mod = Some((outer_mod_decl.file_id, result.docs_source_map.len()));
    }

    let inline_source_map_start = result.docs_source_map.len();
    let mut indent = usize::MAX;
    extend_with_attrs(&mut result, source.value.syntax(), false, &mut indent);
    if let Some(inner_attrs_node) = &inner_attrs_node {
        result.inline_inner_docs_start = Some(TextSize::of(&result.docs));
        extend_with_attrs(&mut result, inner_attrs_node, true, &mut indent);
    }
    result.remove_indent(indent, inline_source_map_start);

    result.remove_last_newline();

    result.shrink_to_fit();

    if result.docs.is_empty() { None } else { Some(Box::new(result)) }
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
                    |attr, _, _, _| match_attr_flags(&mut attr_flags, attr),
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
        let generic_params = GenericParams::new(db, def);
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
                expand_cfg_attr(lifetime.attrs(), &mut cfg_options, |attr, _, _, _| {
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
                expand_cfg_attr(type_or_const.attrs(), &mut cfg_options, |attr, _, _, _| {
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
            |attr, _, _, _| {
                if let Meta::TokenTree { path, tt } = attr
                    && path.is1("cfg")
                    && let cfg =
                        CfgExpr::parse_from_ast(&mut TokenTreeChildren::new(&tt).peekable())
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
                if let Meta::NamedKeyValue { name: Some(name), value: Some(value), .. } = attr
                    && name.text() == "lang"
                    && let Some(value) = ast::String::cast(value)
                    && let Ok(value) = value.value()
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
                if let Meta::TokenTree { path, tt } = attr
                    && path.is1("repr")
                    && let Some(repr) = parse_repr_tt(&tt)
                {
                    match &mut result {
                        Some(existing) => merge_repr(existing, repr),
                        None => result = Some(repr),
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
            if let Meta::TokenTree { path, tt } = attr
                && path.is1("rustc_legacy_const_generics")
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
        let syntax = db.parse(root_file_id).tree();

        let mut cfg_options = None;
        expand_cfg_attr(
            syntax.attrs(),
            || cfg_options.get_or_insert(krate.cfg_options(db)),
            |attr, _, _, _| {
                if let Meta::TokenTree { path, tt } = attr
                    && path.is1("doc")
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
                if let Meta::TokenTree { path, tt } = attr
                    && path.is1("target_feature")
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
                if let Meta::TokenTree { path, tt } = attr
                    && (path.is1("rustc_layout_scalar_valid_range_start")
                        || path.is1("rustc_layout_scalar_valid_range_end"))
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
                    |attr, _, _, _| extract_doc_aliases(&mut result, attr),
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
                    |attr, _, _, _| extract_cfgs(&mut result, attr),
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
                if let Meta::TokenTree { path, tt } = attr
                    && path.is1("doc")
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
        let (source, outer_mod_decl, krate) = attrs_source(db, owner);
        let inner_attrs_node = source.value.inner_attributes_node();
        extract_docs(&|| krate.cfg_options(db), source, outer_mod_decl, inner_attrs_node)
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
            collect_field_attrs(db, variant, |cfg_options, field| {
                extract_docs(&|| cfg_options, field, None, None)
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
                if let Meta::TokenTree { path, tt } = attr
                    && path.segments.len() == 1
                    && matches!(
                        path.segments[0].text(),
                        "proc_macro_derive" | "rustc_builtin_macro"
                    )
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
}

fn merge_repr(this: &mut ReprOptions, other: ReprOptions) {
    let ReprOptions { int, align, pack, flags, field_shuffle_seed: _ } = this;
    flags.insert(other.flags);
    *align = (*align).max(other.align);
    *pack = match (*pack, other.pack) {
        (Some(pack), None) | (None, Some(pack)) => Some(pack),
        _ => (*pack).min(other.pack),
    };
    if other.int.is_some() {
        *int = other.int;
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
    use expect_test::expect;
    use hir_expand::InFile;
    use test_fixture::WithFixture;
    use tt::{TextRange, TextSize};

    use crate::attrs::IsInnerDoc;
    use crate::{attrs::Docs, test_db::TestDB};

    #[test]
    fn docs() {
        let (_db, file_id) = TestDB::with_single_file("");
        let mut docs = Docs {
            docs: String::new(),
            docs_source_map: Vec::new(),
            outline_mod: None,
            inline_file: file_id.into(),
            prefix_len: TextSize::new(0),
            inline_inner_docs_start: None,
            outline_inner_docs_start: None,
        };
        let mut indent = usize::MAX;

        let outer = " foo\n\tbar  baz";
        let mut ast_offset = TextSize::new(123);
        for line in outer.split('\n') {
            docs.extend_with_doc_str(line, ast_offset, &mut indent);
            ast_offset += TextSize::of(line) + TextSize::of("\n");
        }

        docs.inline_inner_docs_start = Some(TextSize::of(&docs.docs));
        ast_offset += TextSize::new(123);
        let inner = " bar \n baz";
        for line in inner.split('\n') {
            docs.extend_with_doc_str(line, ast_offset, &mut indent);
            ast_offset += TextSize::of(line) + TextSize::of("\n");
        }

        assert_eq!(indent, 1);
        expect![[r#"
            [
                DocsSourceMapLine {
                    string_offset: 0,
                    ast_offset: 123,
                },
                DocsSourceMapLine {
                    string_offset: 5,
                    ast_offset: 128,
                },
                DocsSourceMapLine {
                    string_offset: 15,
                    ast_offset: 261,
                },
                DocsSourceMapLine {
                    string_offset: 20,
                    ast_offset: 267,
                },
            ]
        "#]]
        .assert_debug_eq(&docs.docs_source_map);

        docs.remove_indent(indent, 0);

        assert_eq!(docs.inline_inner_docs_start, Some(TextSize::new(13)));

        assert_eq!(docs.docs, "foo\nbar  baz\nbar\nbaz\n");
        expect![[r#"
            [
                DocsSourceMapLine {
                    string_offset: 0,
                    ast_offset: 124,
                },
                DocsSourceMapLine {
                    string_offset: 4,
                    ast_offset: 129,
                },
                DocsSourceMapLine {
                    string_offset: 13,
                    ast_offset: 262,
                },
                DocsSourceMapLine {
                    string_offset: 17,
                    ast_offset: 268,
                },
            ]
        "#]]
        .assert_debug_eq(&docs.docs_source_map);

        docs.append(&docs.clone());
        docs.prepend_str("prefix---");
        assert_eq!(docs.docs, "prefix---foo\nbar  baz\nbar\nbaz\nfoo\nbar  baz\nbar\nbaz\n");
        expect![[r#"
            [
                DocsSourceMapLine {
                    string_offset: 0,
                    ast_offset: 124,
                },
                DocsSourceMapLine {
                    string_offset: 4,
                    ast_offset: 129,
                },
                DocsSourceMapLine {
                    string_offset: 13,
                    ast_offset: 262,
                },
                DocsSourceMapLine {
                    string_offset: 17,
                    ast_offset: 268,
                },
                DocsSourceMapLine {
                    string_offset: 21,
                    ast_offset: 124,
                },
                DocsSourceMapLine {
                    string_offset: 25,
                    ast_offset: 129,
                },
                DocsSourceMapLine {
                    string_offset: 34,
                    ast_offset: 262,
                },
                DocsSourceMapLine {
                    string_offset: 38,
                    ast_offset: 268,
                },
            ]
        "#]]
        .assert_debug_eq(&docs.docs_source_map);

        let range = |start, end| TextRange::new(TextSize::new(start), TextSize::new(end));
        let in_file = |range| InFile::new(file_id.into(), range);
        assert_eq!(docs.find_ast_range(range(0, 2)), None);
        assert_eq!(docs.find_ast_range(range(8, 10)), None);
        assert_eq!(
            docs.find_ast_range(range(9, 10)),
            Some((in_file(range(124, 125)), IsInnerDoc::No))
        );
        assert_eq!(docs.find_ast_range(range(20, 23)), None);
        assert_eq!(
            docs.find_ast_range(range(23, 25)),
            Some((in_file(range(263, 265)), IsInnerDoc::Yes))
        );
    }
}
