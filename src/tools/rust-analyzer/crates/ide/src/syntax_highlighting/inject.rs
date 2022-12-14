//! "Recursive" Syntax highlighting for code in doctests and fixtures.

use std::mem;

use either::Either;
use hir::{InFile, Semantics};
use ide_db::{
    active_parameter::ActiveParameter, base_db::FileId, defs::Definition, rust_doc::is_rust_fence,
    SymbolKind,
};
use syntax::{
    ast::{self, AstNode, IsString, QuoteOffsets},
    AstToken, NodeOrToken, SyntaxNode, TextRange, TextSize,
};

use crate::{
    doc_links::{doc_attributes, extract_definitions_from_docs, resolve_doc_path_for_def},
    syntax_highlighting::{highlights::Highlights, injector::Injector, HighlightConfig},
    Analysis, HlMod, HlRange, HlTag, RootDatabase,
};

pub(super) fn ra_fixture(
    hl: &mut Highlights,
    sema: &Semantics<'_, RootDatabase>,
    config: HighlightConfig,
    literal: &ast::String,
    expanded: &ast::String,
) -> Option<()> {
    let active_parameter = ActiveParameter::at_token(sema, expanded.syntax().clone())?;
    if !active_parameter.ident().map_or(false, |name| name.text().starts_with("ra_fixture")) {
        return None;
    }
    let value = literal.value()?;

    if let Some(range) = literal.open_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    let mut inj = Injector::default();

    let mut text = &*value;
    let mut offset: TextSize = 0.into();

    while !text.is_empty() {
        let marker = "$0";
        let idx = text.find(marker).unwrap_or(text.len());
        let (chunk, next) = text.split_at(idx);
        inj.add(chunk, TextRange::at(offset, TextSize::of(chunk)));

        text = next;
        offset += TextSize::of(chunk);

        if let Some(next) = text.strip_prefix(marker) {
            if let Some(range) = literal.map_range_up(TextRange::at(offset, TextSize::of(marker))) {
                hl.add(HlRange { range, highlight: HlTag::Keyword.into(), binding_hash: None });
            }

            text = next;

            let marker_len = TextSize::of(marker);
            offset += marker_len;
        }
    }

    let (analysis, tmp_file_id) = Analysis::from_single_file(inj.take_text());

    for mut hl_range in analysis
        .highlight(
            HighlightConfig { syntactic_name_ref_highlighting: false, ..config },
            tmp_file_id,
        )
        .unwrap()
    {
        for range in inj.map_range_up(hl_range.range) {
            if let Some(range) = literal.map_range_up(range) {
                hl_range.range = range;
                hl.add(hl_range);
            }
        }
    }

    if let Some(range) = literal.close_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    Some(())
}

const RUSTDOC_FENCE_LENGTH: usize = 3;
const RUSTDOC_FENCES: [&str; 2] = ["```", "~~~"];

/// Injection of syntax highlighting of doctests and intra doc links.
pub(super) fn doc_comment(
    hl: &mut Highlights,
    sema: &Semantics<'_, RootDatabase>,
    config: HighlightConfig,
    src_file_id: FileId,
    node: &SyntaxNode,
) {
    let (attributes, def) = match doc_attributes(sema, node) {
        Some(it) => it,
        None => return,
    };
    let src_file_id = src_file_id.into();

    // Extract intra-doc links and emit highlights for them.
    if let Some((docs, doc_mapping)) = attributes.docs_with_rangemap(sema.db) {
        extract_definitions_from_docs(&docs)
            .into_iter()
            .filter_map(|(range, link, ns)| {
                doc_mapping.map(range).filter(|mapping| mapping.file_id == src_file_id).and_then(
                    |InFile { value: mapped_range, .. }| {
                        Some(mapped_range).zip(resolve_doc_path_for_def(sema.db, def, &link, ns))
                    },
                )
            })
            .for_each(|(range, def)| {
                hl.add(HlRange {
                    range,
                    highlight: module_def_to_hl_tag(def)
                        | HlMod::Documentation
                        | HlMod::Injected
                        | HlMod::IntraDocLink,
                    binding_hash: None,
                })
            });
    }

    // Extract doc-test sources from the docs and calculate highlighting for them.

    let mut inj = Injector::default();
    inj.add_unmapped("fn doctest() {\n");

    let attrs_source_map = attributes.source_map(sema.db);

    let mut is_codeblock = false;
    let mut is_doctest = false;

    let mut new_comments = Vec::new();
    let mut string;

    for attr in attributes.by_key("doc").attrs() {
        let InFile { file_id, value: src } = attrs_source_map.source_of(attr);
        if file_id != src_file_id {
            continue;
        }
        let (line, range) = match &src {
            Either::Left(it) => {
                string = match find_doc_string_in_attr(attr, it) {
                    Some(it) => it,
                    None => continue,
                };
                let text = string.text();
                let text_range = string.syntax().text_range();
                match string.quote_offsets() {
                    Some(QuoteOffsets { contents, .. }) => {
                        (&text[contents - text_range.start()], contents)
                    }
                    None => (text, text_range),
                }
            }
            Either::Right(comment) => {
                let value = comment.prefix().len();
                let range = comment.syntax().text_range();
                (
                    &comment.text()[value..],
                    TextRange::new(range.start() + TextSize::try_from(value).unwrap(), range.end()),
                )
            }
        };

        let mut range_start = range.start();
        for line in line.split('\n') {
            let line_len = TextSize::from(line.len() as u32);
            let prev_range_start = {
                let next_range_start = range_start + line_len + TextSize::from(1);
                mem::replace(&mut range_start, next_range_start)
            };
            let mut pos = TextSize::from(0);

            match RUSTDOC_FENCES.into_iter().find_map(|fence| line.find(fence)) {
                Some(idx) => {
                    is_codeblock = !is_codeblock;
                    // Check whether code is rust by inspecting fence guards
                    let guards = &line[idx + RUSTDOC_FENCE_LENGTH..];
                    let is_rust = is_rust_fence(guards);
                    is_doctest = is_codeblock && is_rust;
                    continue;
                }
                None if !is_doctest => continue,
                None => (),
            }

            // whitespace after comment is ignored
            if let Some(ws) = line[pos.into()..].chars().next().filter(|c| c.is_whitespace()) {
                pos += TextSize::of(ws);
            }
            // lines marked with `#` should be ignored in output, we skip the `#` char
            if line[pos.into()..].starts_with('#') {
                pos += TextSize::of('#');
            }

            new_comments.push(TextRange::at(prev_range_start, pos));
            inj.add(&line[pos.into()..], TextRange::new(pos, line_len) + prev_range_start);
            inj.add_unmapped("\n");
        }
    }

    if new_comments.is_empty() {
        return; // no need to run an analysis on an empty file
    }

    inj.add_unmapped("\n}");

    let (analysis, tmp_file_id) = Analysis::from_single_file(inj.take_text());

    if let Ok(ranges) = analysis.with_db(|db| {
        super::highlight(
            db,
            HighlightConfig { syntactic_name_ref_highlighting: true, ..config },
            tmp_file_id,
            None,
        )
    }) {
        for HlRange { range, highlight, binding_hash } in ranges {
            for range in inj.map_range_up(range) {
                hl.add(HlRange { range, highlight: highlight | HlMod::Injected, binding_hash });
            }
        }
    }

    for range in new_comments {
        hl.add(HlRange {
            range,
            highlight: HlTag::Comment | HlMod::Documentation,
            binding_hash: None,
        });
    }
}

fn find_doc_string_in_attr(attr: &hir::Attr, it: &ast::Attr) -> Option<ast::String> {
    match it.expr() {
        // #[doc = lit]
        Some(ast::Expr::Literal(lit)) => match lit.kind() {
            ast::LiteralKind::String(it) => Some(it),
            _ => None,
        },
        // #[cfg_attr(..., doc = "", ...)]
        None => {
            // We gotta hunt the string token manually here
            let text = attr.string_value()?;
            // FIXME: We just pick the first string literal that has the same text as the doc attribute
            // This means technically we might highlight the wrong one
            it.syntax()
                .descendants_with_tokens()
                .filter_map(NodeOrToken::into_token)
                .filter_map(ast::String::cast)
                .find(|string| {
                    string.text().get(1..string.text().len() - 1).map_or(false, |it| it == text)
                })
        }
        _ => None,
    }
}

fn module_def_to_hl_tag(def: Definition) -> HlTag {
    let symbol = match def {
        Definition::Module(_) => SymbolKind::Module,
        Definition::Function(_) => SymbolKind::Function,
        Definition::Adt(hir::Adt::Struct(_)) => SymbolKind::Struct,
        Definition::Adt(hir::Adt::Enum(_)) => SymbolKind::Enum,
        Definition::Adt(hir::Adt::Union(_)) => SymbolKind::Union,
        Definition::Variant(_) => SymbolKind::Variant,
        Definition::Const(_) => SymbolKind::Const,
        Definition::Static(_) => SymbolKind::Static,
        Definition::Trait(_) => SymbolKind::Trait,
        Definition::TypeAlias(_) => SymbolKind::TypeAlias,
        Definition::BuiltinType(_) => return HlTag::BuiltinType,
        Definition::Macro(_) => SymbolKind::Macro,
        Definition::Field(_) => SymbolKind::Field,
        Definition::SelfType(_) => SymbolKind::Impl,
        Definition::Local(_) => SymbolKind::Local,
        Definition::GenericParam(gp) => match gp {
            hir::GenericParam::TypeParam(_) => SymbolKind::TypeParam,
            hir::GenericParam::ConstParam(_) => SymbolKind::ConstParam,
            hir::GenericParam::LifetimeParam(_) => SymbolKind::LifetimeParam,
        },
        Definition::Label(_) => SymbolKind::Label,
        Definition::BuiltinAttr(_) => SymbolKind::BuiltinAttr,
        Definition::ToolModule(_) => SymbolKind::ToolModule,
        Definition::DeriveHelper(_) => SymbolKind::DeriveHelper,
    };
    HlTag::Symbol(symbol)
}
