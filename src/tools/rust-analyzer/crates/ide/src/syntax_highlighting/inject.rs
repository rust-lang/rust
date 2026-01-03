//! "Recursive" Syntax highlighting for code in doctests and fixtures.

use hir::{EditionedFileId, HirFileId, InFile, Semantics};
use ide_db::{
    SymbolKind, defs::Definition, documentation::Documentation, range_mapper::RangeMapper,
    rust_doc::is_rust_fence,
};
use syntax::{
    SyntaxNode, TextRange, TextSize,
    ast::{self, IsString},
};

use crate::{
    Analysis, HlMod, HlRange, HlTag, RootDatabase,
    doc_links::{doc_attributes, extract_definitions_from_docs, resolve_doc_path_for_def},
    syntax_highlighting::{HighlightConfig, highlights::Highlights},
};

pub(super) fn ra_fixture(
    hl: &mut Highlights,
    sema: &Semantics<'_, RootDatabase>,
    config: &HighlightConfig<'_>,
    literal: &ast::String,
    expanded: &ast::String,
) -> Option<()> {
    let (analysis, fixture_analysis) = Analysis::from_ra_fixture_with_on_cursor(
        sema,
        literal.clone(),
        expanded,
        config.minicore,
        &mut |range| {
            hl.add(HlRange {
                range,
                highlight: HlTag::Keyword | HlMod::Injected,
                binding_hash: None,
            });
        },
    )?;

    if let Some(range) = literal.open_quote_text_range() {
        hl.add(HlRange { range, highlight: HlTag::StringLiteral.into(), binding_hash: None })
    }

    for tmp_file_id in fixture_analysis.files() {
        for mut hl_range in analysis
            .highlight(
                HighlightConfig {
                    syntactic_name_ref_highlighting: false,
                    comments: true,
                    punctuation: true,
                    operator: true,
                    strings: true,
                    specialize_punctuation: config.specialize_punctuation,
                    specialize_operator: config.operator,
                    inject_doc_comment: config.inject_doc_comment,
                    macro_bang: config.macro_bang,
                    // What if there is a fixture inside a fixture? It's fixtures all the way down.
                    // (In fact, we have a fixture inside a fixture in our test suite!)
                    minicore: config.minicore,
                },
                tmp_file_id,
            )
            .unwrap()
        {
            for range in fixture_analysis.map_range_up(tmp_file_id, hl_range.range) {
                hl_range.range = range;
                hl_range.highlight |= HlMod::Injected;
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
    config: &HighlightConfig<'_>,
    src_file_id: EditionedFileId,
    node: &SyntaxNode,
) {
    let (attributes, def) = match doc_attributes(sema, node) {
        Some(it) => it,
        None => return,
    };
    let src_file_id: HirFileId = src_file_id.into();
    let Some(docs) = attributes.hir_docs(sema.db) else { return };

    // Extract intra-doc links and emit highlights for them.
    extract_definitions_from_docs(&Documentation::new_borrowed(docs.docs()))
        .into_iter()
        .filter_map(|(range, link, ns)| {
            docs.find_ast_range(range)
                .filter(|(mapping, _)| mapping.file_id == src_file_id)
                .and_then(|(InFile { value: mapped_range, .. }, is_inner)| {
                    Some(mapped_range)
                        .zip(resolve_doc_path_for_def(sema.db, def, &link, ns, is_inner))
                })
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

    // Extract doc-test sources from the docs and calculate highlighting for them.

    let mut inj = RangeMapper::default();
    inj.add_unmapped("fn doctest() {\n");

    let mut is_codeblock = false;
    let mut is_doctest = false;

    let mut has_doctests = false;

    let mut docs_offset = TextSize::new(0);
    for mut line in docs.docs().split('\n') {
        let mut line_docs_offset = docs_offset;
        docs_offset += TextSize::of(line) + TextSize::of("\n");

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

        // lines marked with `#` should be ignored in output, we skip the `#` char
        if line.starts_with('#') {
            line_docs_offset += TextSize::of("#");
            line = &line["#".len()..];
        }

        let Some((InFile { file_id, value: mapped_range }, _)) =
            docs.find_ast_range(TextRange::at(line_docs_offset, TextSize::of(line)))
        else {
            continue;
        };
        if file_id != src_file_id {
            continue;
        }

        has_doctests = true;
        inj.add(line, mapped_range);
        inj.add_unmapped("\n");
    }

    if !has_doctests {
        return; // no need to run an analysis on an empty file
    }

    inj.add_unmapped("\n}");

    let (analysis, tmp_file_id) = Analysis::from_single_file(inj.take_text());

    if let Ok(ranges) = analysis.with_db(|db| {
        super::highlight(
            db,
            &HighlightConfig {
                syntactic_name_ref_highlighting: true,
                comments: true,
                punctuation: true,
                operator: true,
                strings: true,
                specialize_punctuation: config.specialize_punctuation,
                specialize_operator: config.operator,
                inject_doc_comment: config.inject_doc_comment,
                macro_bang: config.macro_bang,
                minicore: config.minicore,
            },
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
}

fn module_def_to_hl_tag(def: Definition) -> HlTag {
    let symbol = match def {
        Definition::Module(_) | Definition::Crate(_) | Definition::ExternCrateDecl(_) => {
            SymbolKind::Module
        }
        Definition::Function(_) => SymbolKind::Function,
        Definition::Adt(hir::Adt::Struct(_)) => SymbolKind::Struct,
        Definition::Adt(hir::Adt::Enum(_)) => SymbolKind::Enum,
        Definition::Adt(hir::Adt::Union(_)) => SymbolKind::Union,
        Definition::Variant(_) => SymbolKind::Variant,
        Definition::Const(_) => SymbolKind::Const,
        Definition::Static(_) => SymbolKind::Static,
        Definition::Trait(_) => SymbolKind::Trait,
        Definition::TypeAlias(_) => SymbolKind::TypeAlias,
        Definition::BuiltinLifetime(_) => SymbolKind::LifetimeParam,
        Definition::BuiltinType(_) => return HlTag::BuiltinType,
        Definition::Macro(_) => SymbolKind::Macro,
        Definition::Field(_) | Definition::TupleField(_) => SymbolKind::Field,
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
        Definition::InlineAsmRegOrRegClass(_) => SymbolKind::InlineAsmRegOrRegClass,
        Definition::InlineAsmOperand(_) => SymbolKind::Local,
    };
    HlTag::Symbol(symbol)
}
