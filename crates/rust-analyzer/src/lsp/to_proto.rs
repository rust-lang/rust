//! Conversion of rust-analyzer specific types to lsp_types equivalents.
use std::{
    iter::once,
    mem, path,
    sync::atomic::{AtomicU32, Ordering},
};

use ide::{
    Annotation, AnnotationKind, Assist, AssistKind, Cancellable, CompletionItem,
    CompletionItemKind, CompletionRelevance, Documentation, FileId, FileRange, FileSystemEdit,
    Fold, FoldKind, Highlight, HlMod, HlOperator, HlPunct, HlRange, HlTag, Indel,
    InlayFieldsToResolve, InlayHint, InlayHintLabel, InlayHintLabelPart, InlayKind, Markup,
    NavigationTarget, ReferenceCategory, RenameError, Runnable, Severity, SignatureHelp,
    SnippetEdit, SourceChange, StructureNodeKind, SymbolKind, TextEdit, TextRange, TextSize,
};
use ide_db::rust_doc::format_docs;
use itertools::Itertools;
use semver::VersionReq;
use serde_json::to_value;
use vfs::AbsPath;

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::{CallInfoConfig, Config},
    global_state::GlobalStateSnapshot,
    line_index::{LineEndings, LineIndex, PositionEncoding},
    lsp::{
        semantic_tokens::{self, standard_fallback_type},
        utils::invalid_params_error,
        LspError,
    },
    lsp_ext::{self, SnippetTextEdit},
};

pub(crate) fn position(line_index: &LineIndex, offset: TextSize) -> lsp_types::Position {
    let line_col = line_index.index.line_col(offset);
    match line_index.encoding {
        PositionEncoding::Utf8 => lsp_types::Position::new(line_col.line, line_col.col),
        PositionEncoding::Wide(enc) => {
            let line_col = line_index.index.to_wide(enc, line_col).unwrap();
            lsp_types::Position::new(line_col.line, line_col.col)
        }
    }
}

pub(crate) fn range(line_index: &LineIndex, range: TextRange) -> lsp_types::Range {
    let start = position(line_index, range.start());
    let end = position(line_index, range.end());
    lsp_types::Range::new(start, end)
}

pub(crate) fn symbol_kind(symbol_kind: SymbolKind) -> lsp_types::SymbolKind {
    match symbol_kind {
        SymbolKind::Function => lsp_types::SymbolKind::FUNCTION,
        SymbolKind::Struct => lsp_types::SymbolKind::STRUCT,
        SymbolKind::Enum => lsp_types::SymbolKind::ENUM,
        SymbolKind::Variant => lsp_types::SymbolKind::ENUM_MEMBER,
        SymbolKind::Trait | SymbolKind::TraitAlias => lsp_types::SymbolKind::INTERFACE,
        SymbolKind::Macro
        | SymbolKind::ProcMacro
        | SymbolKind::BuiltinAttr
        | SymbolKind::Attribute
        | SymbolKind::Derive
        | SymbolKind::DeriveHelper => lsp_types::SymbolKind::FUNCTION,
        SymbolKind::Module | SymbolKind::ToolModule => lsp_types::SymbolKind::MODULE,
        SymbolKind::TypeAlias | SymbolKind::TypeParam | SymbolKind::SelfType => {
            lsp_types::SymbolKind::TYPE_PARAMETER
        }
        SymbolKind::Field => lsp_types::SymbolKind::FIELD,
        SymbolKind::Static => lsp_types::SymbolKind::CONSTANT,
        SymbolKind::Const => lsp_types::SymbolKind::CONSTANT,
        SymbolKind::ConstParam => lsp_types::SymbolKind::CONSTANT,
        SymbolKind::Impl => lsp_types::SymbolKind::OBJECT,
        SymbolKind::Local
        | SymbolKind::SelfParam
        | SymbolKind::LifetimeParam
        | SymbolKind::ValueParam
        | SymbolKind::Label => lsp_types::SymbolKind::VARIABLE,
        SymbolKind::Union => lsp_types::SymbolKind::STRUCT,
    }
}

pub(crate) fn structure_node_kind(kind: StructureNodeKind) -> lsp_types::SymbolKind {
    match kind {
        StructureNodeKind::SymbolKind(symbol) => symbol_kind(symbol),
        StructureNodeKind::Region => lsp_types::SymbolKind::NAMESPACE,
    }
}

pub(crate) fn document_highlight_kind(
    category: ReferenceCategory,
) -> Option<lsp_types::DocumentHighlightKind> {
    match category {
        ReferenceCategory::Read => Some(lsp_types::DocumentHighlightKind::READ),
        ReferenceCategory::Write => Some(lsp_types::DocumentHighlightKind::WRITE),
        ReferenceCategory::Import => None,
        ReferenceCategory::Test => None,
    }
}

pub(crate) fn diagnostic_severity(severity: Severity) -> lsp_types::DiagnosticSeverity {
    match severity {
        Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
        Severity::Warning => lsp_types::DiagnosticSeverity::WARNING,
        Severity::WeakWarning => lsp_types::DiagnosticSeverity::HINT,
        // unreachable
        Severity::Allow => lsp_types::DiagnosticSeverity::INFORMATION,
    }
}

pub(crate) fn documentation(documentation: Documentation) -> lsp_types::Documentation {
    let value = format_docs(&documentation);
    let markup_content = lsp_types::MarkupContent { kind: lsp_types::MarkupKind::Markdown, value };
    lsp_types::Documentation::MarkupContent(markup_content)
}

pub(crate) fn completion_item_kind(
    completion_item_kind: CompletionItemKind,
) -> lsp_types::CompletionItemKind {
    match completion_item_kind {
        CompletionItemKind::Binding => lsp_types::CompletionItemKind::VARIABLE,
        CompletionItemKind::BuiltinType => lsp_types::CompletionItemKind::STRUCT,
        CompletionItemKind::InferredType => lsp_types::CompletionItemKind::SNIPPET,
        CompletionItemKind::Keyword => lsp_types::CompletionItemKind::KEYWORD,
        CompletionItemKind::Method => lsp_types::CompletionItemKind::METHOD,
        CompletionItemKind::Snippet => lsp_types::CompletionItemKind::SNIPPET,
        CompletionItemKind::UnresolvedReference => lsp_types::CompletionItemKind::REFERENCE,
        CompletionItemKind::Expression => lsp_types::CompletionItemKind::SNIPPET,
        CompletionItemKind::SymbolKind(symbol) => match symbol {
            SymbolKind::Attribute => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::Const => lsp_types::CompletionItemKind::CONSTANT,
            SymbolKind::ConstParam => lsp_types::CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::Derive => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::DeriveHelper => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::Enum => lsp_types::CompletionItemKind::ENUM,
            SymbolKind::Field => lsp_types::CompletionItemKind::FIELD,
            SymbolKind::Function => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::Impl => lsp_types::CompletionItemKind::TEXT,
            SymbolKind::Label => lsp_types::CompletionItemKind::VARIABLE,
            SymbolKind::LifetimeParam => lsp_types::CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::Local => lsp_types::CompletionItemKind::VARIABLE,
            SymbolKind::Macro => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::ProcMacro => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::Module => lsp_types::CompletionItemKind::MODULE,
            SymbolKind::SelfParam => lsp_types::CompletionItemKind::VALUE,
            SymbolKind::SelfType => lsp_types::CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::Static => lsp_types::CompletionItemKind::VALUE,
            SymbolKind::Struct => lsp_types::CompletionItemKind::STRUCT,
            SymbolKind::Trait => lsp_types::CompletionItemKind::INTERFACE,
            SymbolKind::TraitAlias => lsp_types::CompletionItemKind::INTERFACE,
            SymbolKind::TypeAlias => lsp_types::CompletionItemKind::STRUCT,
            SymbolKind::TypeParam => lsp_types::CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::Union => lsp_types::CompletionItemKind::STRUCT,
            SymbolKind::ValueParam => lsp_types::CompletionItemKind::VALUE,
            SymbolKind::Variant => lsp_types::CompletionItemKind::ENUM_MEMBER,
            SymbolKind::BuiltinAttr => lsp_types::CompletionItemKind::FUNCTION,
            SymbolKind::ToolModule => lsp_types::CompletionItemKind::MODULE,
        },
    }
}

pub(crate) fn text_edit(line_index: &LineIndex, indel: Indel) -> lsp_types::TextEdit {
    let range = range(line_index, indel.delete);
    let new_text = match line_index.endings {
        LineEndings::Unix => indel.insert,
        LineEndings::Dos => indel.insert.replace('\n', "\r\n"),
    };
    lsp_types::TextEdit { range, new_text }
}

pub(crate) fn completion_text_edit(
    line_index: &LineIndex,
    insert_replace_support: Option<lsp_types::Position>,
    indel: Indel,
) -> lsp_types::CompletionTextEdit {
    let text_edit = text_edit(line_index, indel);
    match insert_replace_support {
        Some(cursor_pos) => lsp_types::InsertReplaceEdit {
            new_text: text_edit.new_text,
            insert: lsp_types::Range { start: text_edit.range.start, end: cursor_pos },
            replace: text_edit.range,
        }
        .into(),
        None => text_edit.into(),
    }
}

pub(crate) fn snippet_text_edit(
    line_index: &LineIndex,
    is_snippet: bool,
    indel: Indel,
) -> lsp_ext::SnippetTextEdit {
    let text_edit = text_edit(line_index, indel);
    let insert_text_format =
        if is_snippet { Some(lsp_types::InsertTextFormat::SNIPPET) } else { None };
    lsp_ext::SnippetTextEdit {
        range: text_edit.range,
        new_text: text_edit.new_text,
        insert_text_format,
        annotation_id: None,
    }
}

pub(crate) fn text_edit_vec(
    line_index: &LineIndex,
    text_edit: TextEdit,
) -> Vec<lsp_types::TextEdit> {
    text_edit.into_iter().map(|indel| self::text_edit(line_index, indel)).collect()
}

pub(crate) fn snippet_text_edit_vec(
    line_index: &LineIndex,
    is_snippet: bool,
    text_edit: TextEdit,
) -> Vec<lsp_ext::SnippetTextEdit> {
    text_edit
        .into_iter()
        .map(|indel| self::snippet_text_edit(line_index, is_snippet, indel))
        .collect()
}

pub(crate) fn completion_items(
    config: &Config,
    line_index: &LineIndex,
    tdpp: lsp_types::TextDocumentPositionParams,
    items: Vec<CompletionItem>,
) -> Vec<lsp_types::CompletionItem> {
    let max_relevance = items.iter().map(|it| it.relevance.score()).max().unwrap_or_default();
    let mut res = Vec::with_capacity(items.len());
    for item in items {
        completion_item(&mut res, config, line_index, &tdpp, max_relevance, item);
    }

    if let Some(limit) = config.completion().limit {
        res.sort_by(|item1, item2| item1.sort_text.cmp(&item2.sort_text));
        res.truncate(limit);
    }

    res
}

fn completion_item(
    acc: &mut Vec<lsp_types::CompletionItem>,
    config: &Config,
    line_index: &LineIndex,
    tdpp: &lsp_types::TextDocumentPositionParams,
    max_relevance: u32,
    item: CompletionItem,
) {
    let insert_replace_support = config.insert_replace_support().then_some(tdpp.position);
    let ref_match = item.ref_match();
    let lookup = item.lookup().to_owned();

    let mut additional_text_edits = Vec::new();

    // LSP does not allow arbitrary edits in completion, so we have to do a
    // non-trivial mapping here.
    let text_edit = {
        let mut text_edit = None;
        let source_range = item.source_range;
        for indel in item.text_edit {
            if indel.delete.contains_range(source_range) {
                // Extract this indel as the main edit
                text_edit = Some(if indel.delete == source_range {
                    self::completion_text_edit(line_index, insert_replace_support, indel.clone())
                } else {
                    assert!(source_range.end() == indel.delete.end());
                    let range1 = TextRange::new(indel.delete.start(), source_range.start());
                    let range2 = source_range;
                    let indel1 = Indel::delete(range1);
                    let indel2 = Indel::replace(range2, indel.insert.clone());
                    additional_text_edits.push(self::text_edit(line_index, indel1));
                    self::completion_text_edit(line_index, insert_replace_support, indel2)
                })
            } else {
                assert!(source_range.intersect(indel.delete).is_none());
                let text_edit = self::text_edit(line_index, indel.clone());
                additional_text_edits.push(text_edit);
            }
        }
        text_edit.unwrap()
    };

    let insert_text_format = item.is_snippet.then_some(lsp_types::InsertTextFormat::SNIPPET);
    let tags = item.deprecated.then(|| vec![lsp_types::CompletionItemTag::DEPRECATED]);
    let command = if item.trigger_call_info && config.client_commands().trigger_parameter_hints {
        Some(command::trigger_parameter_hints())
    } else {
        None
    };

    let mut lsp_item = lsp_types::CompletionItem {
        label: item.label.to_string(),
        detail: item.detail,
        filter_text: Some(lookup),
        kind: Some(completion_item_kind(item.kind)),
        text_edit: Some(text_edit),
        additional_text_edits: Some(additional_text_edits),
        documentation: item.documentation.map(documentation),
        deprecated: Some(item.deprecated),
        tags,
        command,
        insert_text_format,
        ..Default::default()
    };

    if config.completion_label_details_support() {
        lsp_item.label_details = Some(lsp_types::CompletionItemLabelDetails {
            detail: item.label_detail.as_ref().map(ToString::to_string),
            description: lsp_item.detail.clone(),
        });
    } else if let Some(label_detail) = item.label_detail {
        lsp_item.label.push_str(label_detail.as_str());
    }

    set_score(&mut lsp_item, max_relevance, item.relevance);

    if config.completion().enable_imports_on_the_fly && !item.import_to_add.is_empty() {
        let imports = item
            .import_to_add
            .into_iter()
            .map(|(import_path, import_name)| lsp_ext::CompletionImport {
                full_import_path: import_path,
                imported_name: import_name,
            })
            .collect::<Vec<_>>();
        if !imports.is_empty() {
            let data = lsp_ext::CompletionResolveData { position: tdpp.clone(), imports };
            lsp_item.data = Some(to_value(data).unwrap());
        }
    }

    if let Some((label, indel, relevance)) = ref_match {
        let mut lsp_item_with_ref = lsp_types::CompletionItem { label, ..lsp_item.clone() };
        lsp_item_with_ref
            .additional_text_edits
            .get_or_insert_with(Default::default)
            .push(self::text_edit(line_index, indel));
        set_score(&mut lsp_item_with_ref, max_relevance, relevance);
        acc.push(lsp_item_with_ref);
    };

    acc.push(lsp_item);

    fn set_score(
        res: &mut lsp_types::CompletionItem,
        max_relevance: u32,
        relevance: CompletionRelevance,
    ) {
        if relevance.is_relevant() && relevance.score() == max_relevance {
            res.preselect = Some(true);
        }
        // The relevance needs to be inverted to come up with a sort score
        // because the client will sort ascending.
        let sort_score = relevance.score() ^ 0xFF_FF_FF_FF;
        // Zero pad the string to ensure values can be properly sorted
        // by the client. Hex format is used because it is easier to
        // visually compare very large values, which the sort text
        // tends to be since it is the opposite of the score.
        res.sort_text = Some(format!("{sort_score:08x}"));
    }
}

pub(crate) fn signature_help(
    call_info: SignatureHelp,
    config: CallInfoConfig,
    label_offsets: bool,
) -> lsp_types::SignatureHelp {
    let (label, parameters) = match (config.params_only, label_offsets) {
        (concise, false) => {
            let params = call_info
                .parameter_labels()
                .map(|label| lsp_types::ParameterInformation {
                    label: lsp_types::ParameterLabel::Simple(label.to_owned()),
                    documentation: None,
                })
                .collect::<Vec<_>>();
            let label =
                if concise { call_info.parameter_labels().join(", ") } else { call_info.signature };
            (label, params)
        }
        (false, true) => {
            let params = call_info
                .parameter_ranges()
                .iter()
                .map(|it| {
                    let start = call_info.signature[..it.start().into()].chars().count() as u32;
                    let end = call_info.signature[..it.end().into()].chars().count() as u32;
                    [start, end]
                })
                .map(|label_offsets| lsp_types::ParameterInformation {
                    label: lsp_types::ParameterLabel::LabelOffsets(label_offsets),
                    documentation: None,
                })
                .collect::<Vec<_>>();
            (call_info.signature, params)
        }
        (true, true) => {
            let mut params = Vec::new();
            let mut label = String::new();
            let mut first = true;
            for param in call_info.parameter_labels() {
                if !first {
                    label.push_str(", ");
                }
                first = false;
                let start = label.chars().count() as u32;
                label.push_str(param);
                let end = label.chars().count() as u32;
                params.push(lsp_types::ParameterInformation {
                    label: lsp_types::ParameterLabel::LabelOffsets([start, end]),
                    documentation: None,
                });
            }

            (label, params)
        }
    };

    let documentation = call_info.doc.filter(|_| config.docs).map(|doc| {
        lsp_types::Documentation::MarkupContent(lsp_types::MarkupContent {
            kind: lsp_types::MarkupKind::Markdown,
            value: format_docs(&doc),
        })
    });

    let active_parameter = call_info.active_parameter.map(|it| it as u32);

    let signature = lsp_types::SignatureInformation {
        label,
        documentation,
        parameters: Some(parameters),
        active_parameter,
    };
    lsp_types::SignatureHelp {
        signatures: vec![signature],
        active_signature: Some(0),
        active_parameter,
    }
}

pub(crate) fn inlay_hint(
    snap: &GlobalStateSnapshot,
    fields_to_resolve: &InlayFieldsToResolve,
    line_index: &LineIndex,
    file_id: FileId,
    inlay_hint: InlayHint,
) -> Cancellable<lsp_types::InlayHint> {
    let needs_resolve = inlay_hint.needs_resolve;
    let (label, tooltip, mut something_to_resolve) =
        inlay_hint_label(snap, fields_to_resolve, needs_resolve, inlay_hint.label)?;

    let text_edits = if snap
        .config
        .visual_studio_code_version()
        // https://github.com/microsoft/vscode/issues/193124
        .map_or(true, |version| VersionReq::parse(">=1.86.0").unwrap().matches(version))
        && needs_resolve
        && fields_to_resolve.resolve_text_edits
    {
        something_to_resolve |= inlay_hint.text_edit.is_some();
        None
    } else {
        inlay_hint.text_edit.map(|it| text_edit_vec(line_index, it))
    };

    let data = if needs_resolve && something_to_resolve {
        Some(to_value(lsp_ext::InlayHintResolveData { file_id: file_id.index() }).unwrap())
    } else {
        None
    };

    Ok(lsp_types::InlayHint {
        position: match inlay_hint.position {
            ide::InlayHintPosition::Before => position(line_index, inlay_hint.range.start()),
            ide::InlayHintPosition::After => position(line_index, inlay_hint.range.end()),
        },
        padding_left: Some(inlay_hint.pad_left),
        padding_right: Some(inlay_hint.pad_right),
        kind: match inlay_hint.kind {
            InlayKind::Parameter => Some(lsp_types::InlayHintKind::PARAMETER),
            InlayKind::Type | InlayKind::Chaining => Some(lsp_types::InlayHintKind::TYPE),
            _ => None,
        },
        text_edits,
        data,
        tooltip,
        label,
    })
}

fn inlay_hint_label(
    snap: &GlobalStateSnapshot,
    fields_to_resolve: &InlayFieldsToResolve,
    needs_resolve: bool,
    mut label: InlayHintLabel,
) -> Cancellable<(lsp_types::InlayHintLabel, Option<lsp_types::InlayHintTooltip>, bool)> {
    let mut something_to_resolve = false;
    let (label, tooltip) = match &*label.parts {
        [InlayHintLabelPart { linked_location: None, .. }] => {
            let InlayHintLabelPart { text, tooltip, .. } = label.parts.pop().unwrap();
            let hint_tooltip = if needs_resolve && fields_to_resolve.resolve_hint_tooltip {
                something_to_resolve |= tooltip.is_some();
                None
            } else {
                match tooltip {
                    Some(ide::InlayTooltip::String(s)) => {
                        Some(lsp_types::InlayHintTooltip::String(s))
                    }
                    Some(ide::InlayTooltip::Markdown(s)) => {
                        Some(lsp_types::InlayHintTooltip::MarkupContent(lsp_types::MarkupContent {
                            kind: lsp_types::MarkupKind::Markdown,
                            value: s,
                        }))
                    }
                    None => None,
                }
            };
            (lsp_types::InlayHintLabel::String(text), hint_tooltip)
        }
        _ => {
            let parts = label
                .parts
                .into_iter()
                .map(|part| {
                    let tooltip = if needs_resolve && fields_to_resolve.resolve_label_tooltip {
                        something_to_resolve |= part.tooltip.is_some();
                        None
                    } else {
                        match part.tooltip {
                            Some(ide::InlayTooltip::String(s)) => {
                                Some(lsp_types::InlayHintLabelPartTooltip::String(s))
                            }
                            Some(ide::InlayTooltip::Markdown(s)) => {
                                Some(lsp_types::InlayHintLabelPartTooltip::MarkupContent(
                                    lsp_types::MarkupContent {
                                        kind: lsp_types::MarkupKind::Markdown,
                                        value: s,
                                    },
                                ))
                            }
                            None => None,
                        }
                    };
                    let location = if needs_resolve && fields_to_resolve.resolve_label_location {
                        something_to_resolve |= part.linked_location.is_some();
                        None
                    } else {
                        part.linked_location.map(|range| location(snap, range)).transpose()?
                    };
                    Ok(lsp_types::InlayHintLabelPart {
                        value: part.text,
                        tooltip,
                        location,
                        command: None,
                    })
                })
                .collect::<Cancellable<_>>()?;
            (lsp_types::InlayHintLabel::LabelParts(parts), None)
        }
    };
    Ok((label, tooltip, something_to_resolve))
}

static TOKEN_RESULT_COUNTER: AtomicU32 = AtomicU32::new(1);

pub(crate) fn semantic_tokens(
    text: &str,
    line_index: &LineIndex,
    highlights: Vec<HlRange>,
    semantics_tokens_augments_syntax_tokens: bool,
    non_standard_tokens: bool,
) -> lsp_types::SemanticTokens {
    let id = TOKEN_RESULT_COUNTER.fetch_add(1, Ordering::SeqCst).to_string();
    let mut builder = semantic_tokens::SemanticTokensBuilder::new(id);

    for highlight_range in highlights {
        if highlight_range.highlight.is_empty() {
            continue;
        }

        if semantics_tokens_augments_syntax_tokens {
            match highlight_range.highlight.tag {
                HlTag::BoolLiteral
                | HlTag::ByteLiteral
                | HlTag::CharLiteral
                | HlTag::Comment
                | HlTag::Keyword
                | HlTag::NumericLiteral
                | HlTag::Operator(_)
                | HlTag::Punctuation(_)
                | HlTag::StringLiteral
                | HlTag::None
                    if highlight_range.highlight.mods.is_empty() =>
                {
                    continue
                }
                _ => (),
            }
        }

        let (mut ty, mut mods) = semantic_token_type_and_modifiers(highlight_range.highlight);

        if !non_standard_tokens {
            ty = match standard_fallback_type(ty) {
                Some(ty) => ty,
                None => continue,
            };
            mods.standard_fallback();
        }
        let token_index = semantic_tokens::type_index(ty);
        let modifier_bitset = mods.0;

        for mut text_range in line_index.index.lines(highlight_range.range) {
            if text[text_range].ends_with('\n') {
                text_range =
                    TextRange::new(text_range.start(), text_range.end() - TextSize::of('\n'));
            }
            let range = range(line_index, text_range);
            builder.push(range, token_index, modifier_bitset);
        }
    }

    builder.build()
}

pub(crate) fn semantic_token_delta(
    previous: &lsp_types::SemanticTokens,
    current: &lsp_types::SemanticTokens,
) -> lsp_types::SemanticTokensDelta {
    let result_id = current.result_id.clone();
    let edits = semantic_tokens::diff_tokens(&previous.data, &current.data);
    lsp_types::SemanticTokensDelta { result_id, edits }
}

fn semantic_token_type_and_modifiers(
    highlight: Highlight,
) -> (lsp_types::SemanticTokenType, semantic_tokens::ModifierSet) {
    let mut mods = semantic_tokens::ModifierSet::default();
    let type_ = match highlight.tag {
        HlTag::Symbol(symbol) => match symbol {
            SymbolKind::Attribute => semantic_tokens::DECORATOR,
            SymbolKind::Derive => semantic_tokens::DERIVE,
            SymbolKind::DeriveHelper => semantic_tokens::DERIVE_HELPER,
            SymbolKind::Module => semantic_tokens::NAMESPACE,
            SymbolKind::Impl => semantic_tokens::TYPE_ALIAS,
            SymbolKind::Field => semantic_tokens::PROPERTY,
            SymbolKind::TypeParam => semantic_tokens::TYPE_PARAMETER,
            SymbolKind::ConstParam => semantic_tokens::CONST_PARAMETER,
            SymbolKind::LifetimeParam => semantic_tokens::LIFETIME,
            SymbolKind::Label => semantic_tokens::LABEL,
            SymbolKind::ValueParam => semantic_tokens::PARAMETER,
            SymbolKind::SelfParam => semantic_tokens::SELF_KEYWORD,
            SymbolKind::SelfType => semantic_tokens::SELF_TYPE_KEYWORD,
            SymbolKind::Local => semantic_tokens::VARIABLE,
            SymbolKind::Function => {
                if highlight.mods.contains(HlMod::Associated) {
                    semantic_tokens::METHOD
                } else {
                    semantic_tokens::FUNCTION
                }
            }
            SymbolKind::Const => {
                mods |= semantic_tokens::CONSTANT;
                mods |= semantic_tokens::STATIC;
                semantic_tokens::VARIABLE
            }
            SymbolKind::Static => {
                mods |= semantic_tokens::STATIC;
                semantic_tokens::VARIABLE
            }
            SymbolKind::Struct => semantic_tokens::STRUCT,
            SymbolKind::Enum => semantic_tokens::ENUM,
            SymbolKind::Variant => semantic_tokens::ENUM_MEMBER,
            SymbolKind::Union => semantic_tokens::UNION,
            SymbolKind::TypeAlias => semantic_tokens::TYPE_ALIAS,
            SymbolKind::Trait => semantic_tokens::INTERFACE,
            SymbolKind::TraitAlias => semantic_tokens::INTERFACE,
            SymbolKind::Macro => semantic_tokens::MACRO,
            SymbolKind::ProcMacro => semantic_tokens::PROC_MACRO,
            SymbolKind::BuiltinAttr => semantic_tokens::BUILTIN_ATTRIBUTE,
            SymbolKind::ToolModule => semantic_tokens::TOOL_MODULE,
        },
        HlTag::AttributeBracket => semantic_tokens::ATTRIBUTE_BRACKET,
        HlTag::BoolLiteral => semantic_tokens::BOOLEAN,
        HlTag::BuiltinType => semantic_tokens::BUILTIN_TYPE,
        HlTag::ByteLiteral | HlTag::NumericLiteral => semantic_tokens::NUMBER,
        HlTag::CharLiteral => semantic_tokens::CHAR,
        HlTag::Comment => semantic_tokens::COMMENT,
        HlTag::EscapeSequence => semantic_tokens::ESCAPE_SEQUENCE,
        HlTag::InvalidEscapeSequence => semantic_tokens::INVALID_ESCAPE_SEQUENCE,
        HlTag::FormatSpecifier => semantic_tokens::FORMAT_SPECIFIER,
        HlTag::Keyword => semantic_tokens::KEYWORD,
        HlTag::None => semantic_tokens::GENERIC,
        HlTag::Operator(op) => match op {
            HlOperator::Bitwise => semantic_tokens::BITWISE,
            HlOperator::Arithmetic => semantic_tokens::ARITHMETIC,
            HlOperator::Logical => semantic_tokens::LOGICAL,
            HlOperator::Comparison => semantic_tokens::COMPARISON,
            HlOperator::Other => semantic_tokens::OPERATOR,
        },
        HlTag::StringLiteral => semantic_tokens::STRING,
        HlTag::UnresolvedReference => semantic_tokens::UNRESOLVED_REFERENCE,
        HlTag::Punctuation(punct) => match punct {
            HlPunct::Bracket => semantic_tokens::BRACKET,
            HlPunct::Brace => semantic_tokens::BRACE,
            HlPunct::Parenthesis => semantic_tokens::PARENTHESIS,
            HlPunct::Angle => semantic_tokens::ANGLE,
            HlPunct::Comma => semantic_tokens::COMMA,
            HlPunct::Dot => semantic_tokens::DOT,
            HlPunct::Colon => semantic_tokens::COLON,
            HlPunct::Semi => semantic_tokens::SEMICOLON,
            HlPunct::Other => semantic_tokens::PUNCTUATION,
            HlPunct::MacroBang => semantic_tokens::MACRO_BANG,
        },
    };

    for modifier in highlight.mods.iter() {
        let modifier = match modifier {
            HlMod::Associated => continue,
            HlMod::Async => semantic_tokens::ASYNC,
            HlMod::Attribute => semantic_tokens::ATTRIBUTE_MODIFIER,
            HlMod::Callable => semantic_tokens::CALLABLE,
            HlMod::Consuming => semantic_tokens::CONSUMING,
            HlMod::ControlFlow => semantic_tokens::CONTROL_FLOW,
            HlMod::CrateRoot => semantic_tokens::CRATE_ROOT,
            HlMod::DefaultLibrary => semantic_tokens::DEFAULT_LIBRARY,
            HlMod::Definition => semantic_tokens::DECLARATION,
            HlMod::Documentation => semantic_tokens::DOCUMENTATION,
            HlMod::Injected => semantic_tokens::INJECTED,
            HlMod::IntraDocLink => semantic_tokens::INTRA_DOC_LINK,
            HlMod::Library => semantic_tokens::LIBRARY,
            HlMod::Macro => semantic_tokens::MACRO_MODIFIER,
            HlMod::ProcMacro => semantic_tokens::PROC_MACRO_MODIFIER,
            HlMod::Mutable => semantic_tokens::MUTABLE,
            HlMod::Public => semantic_tokens::PUBLIC,
            HlMod::Reference => semantic_tokens::REFERENCE,
            HlMod::Static => semantic_tokens::STATIC,
            HlMod::Trait => semantic_tokens::TRAIT_MODIFIER,
            HlMod::Unsafe => semantic_tokens::UNSAFE,
        };
        mods |= modifier;
    }

    (type_, mods)
}

pub(crate) fn folding_range(
    text: &str,
    line_index: &LineIndex,
    line_folding_only: bool,
    fold: Fold,
) -> lsp_types::FoldingRange {
    let kind = match fold.kind {
        FoldKind::Comment => Some(lsp_types::FoldingRangeKind::Comment),
        FoldKind::Imports => Some(lsp_types::FoldingRangeKind::Imports),
        FoldKind::Region => Some(lsp_types::FoldingRangeKind::Region),
        FoldKind::Mods
        | FoldKind::Block
        | FoldKind::ArgList
        | FoldKind::Consts
        | FoldKind::Statics
        | FoldKind::WhereClause
        | FoldKind::ReturnType
        | FoldKind::Array
        | FoldKind::MatchArm => None,
    };

    let range = range(line_index, fold.range);

    if line_folding_only {
        // Clients with line_folding_only == true (such as VSCode) will fold the whole end line
        // even if it contains text not in the folding range. To prevent that we exclude
        // range.end.line from the folding region if there is more text after range.end
        // on the same line.
        let has_more_text_on_end_line = text[TextRange::new(fold.range.end(), TextSize::of(text))]
            .chars()
            .take_while(|it| *it != '\n')
            .any(|it| !it.is_whitespace());

        let end_line = if has_more_text_on_end_line {
            range.end.line.saturating_sub(1)
        } else {
            range.end.line
        };

        lsp_types::FoldingRange {
            start_line: range.start.line,
            start_character: None,
            end_line,
            end_character: None,
            kind,
            collapsed_text: None,
        }
    } else {
        lsp_types::FoldingRange {
            start_line: range.start.line,
            start_character: Some(range.start.character),
            end_line: range.end.line,
            end_character: Some(range.end.character),
            kind,
            collapsed_text: None,
        }
    }
}

pub(crate) fn url(snap: &GlobalStateSnapshot, file_id: FileId) -> lsp_types::Url {
    snap.file_id_to_url(file_id)
}

/// Returns a `Url` object from a given path, will lowercase drive letters if present.
/// This will only happen when processing windows paths.
///
/// When processing non-windows path, this is essentially the same as `Url::from_file_path`.
pub(crate) fn url_from_abs_path(path: &AbsPath) -> lsp_types::Url {
    let url = lsp_types::Url::from_file_path(path).unwrap();
    match path.as_ref().components().next() {
        Some(path::Component::Prefix(prefix))
            if matches!(prefix.kind(), path::Prefix::Disk(_) | path::Prefix::VerbatimDisk(_)) =>
        {
            // Need to lowercase driver letter
        }
        _ => return url,
    }

    let driver_letter_range = {
        let (scheme, drive_letter, _rest) = match url.as_str().splitn(3, ':').collect_tuple() {
            Some(it) => it,
            None => return url,
        };
        let start = scheme.len() + ':'.len_utf8();
        start..(start + drive_letter.len())
    };

    // Note: lowercasing the `path` itself doesn't help, the `Url::parse`
    // machinery *also* canonicalizes the drive letter. So, just massage the
    // string in place.
    let mut url: String = url.into();
    url[driver_letter_range].make_ascii_lowercase();
    lsp_types::Url::parse(&url).unwrap()
}

pub(crate) fn optional_versioned_text_document_identifier(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
) -> lsp_types::OptionalVersionedTextDocumentIdentifier {
    let url = url(snap, file_id);
    let version = snap.url_file_version(&url);
    lsp_types::OptionalVersionedTextDocumentIdentifier { uri: url, version }
}

pub(crate) fn location(
    snap: &GlobalStateSnapshot,
    frange: FileRange,
) -> Cancellable<lsp_types::Location> {
    let url = url(snap, frange.file_id);
    let line_index = snap.file_line_index(frange.file_id)?;
    let range = range(&line_index, frange.range);
    let loc = lsp_types::Location::new(url, range);
    Ok(loc)
}

/// Prefer using `location_link`, if the client has the cap.
pub(crate) fn location_from_nav(
    snap: &GlobalStateSnapshot,
    nav: NavigationTarget,
) -> Cancellable<lsp_types::Location> {
    let url = url(snap, nav.file_id);
    let line_index = snap.file_line_index(nav.file_id)?;
    let range = range(&line_index, nav.focus_or_full_range());
    let loc = lsp_types::Location::new(url, range);
    Ok(loc)
}

pub(crate) fn location_link(
    snap: &GlobalStateSnapshot,
    src: Option<FileRange>,
    target: NavigationTarget,
) -> Cancellable<lsp_types::LocationLink> {
    let origin_selection_range = match src {
        Some(src) => {
            let line_index = snap.file_line_index(src.file_id)?;
            let range = range(&line_index, src.range);
            Some(range)
        }
        None => None,
    };
    let (target_uri, target_range, target_selection_range) = location_info(snap, target)?;
    let res = lsp_types::LocationLink {
        origin_selection_range,
        target_uri,
        target_range,
        target_selection_range,
    };
    Ok(res)
}

fn location_info(
    snap: &GlobalStateSnapshot,
    target: NavigationTarget,
) -> Cancellable<(lsp_types::Url, lsp_types::Range, lsp_types::Range)> {
    let line_index = snap.file_line_index(target.file_id)?;

    let target_uri = url(snap, target.file_id);
    let target_range = range(&line_index, target.full_range);
    let target_selection_range =
        target.focus_range.map(|it| range(&line_index, it)).unwrap_or(target_range);
    Ok((target_uri, target_range, target_selection_range))
}

pub(crate) fn goto_definition_response(
    snap: &GlobalStateSnapshot,
    src: Option<FileRange>,
    targets: Vec<NavigationTarget>,
) -> Cancellable<lsp_types::GotoDefinitionResponse> {
    if snap.config.location_link() {
        let links = targets
            .into_iter()
            .unique_by(|nav| (nav.file_id, nav.full_range, nav.focus_range))
            .map(|nav| location_link(snap, src, nav))
            .collect::<Cancellable<Vec<_>>>()?;
        Ok(links.into())
    } else {
        let locations = targets
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .unique()
            .map(|range| location(snap, range))
            .collect::<Cancellable<Vec<_>>>()?;
        Ok(locations.into())
    }
}

fn outside_workspace_annotation_id() -> String {
    String::from("OutsideWorkspace")
}

fn merge_text_and_snippet_edits(
    line_index: &LineIndex,
    edit: TextEdit,
    snippet_edit: SnippetEdit,
) -> Vec<SnippetTextEdit> {
    let mut edits: Vec<SnippetTextEdit> = vec![];
    let mut snippets = snippet_edit.into_edit_ranges().into_iter().peekable();
    let text_edits = edit.into_iter();
    // offset to go from the final source location to the original source location
    let mut source_text_offset = 0i32;

    let offset_range = |range: TextRange, offset: i32| -> TextRange {
        // map the snippet range from the target location into the original source location
        let start = u32::from(range.start()).checked_add_signed(offset).unwrap_or(0);
        let end = u32::from(range.end()).checked_add_signed(offset).unwrap_or(0);

        TextRange::new(start.into(), end.into())
    };

    for current_indel in text_edits {
        let new_range = {
            let insert_len =
                TextSize::try_from(current_indel.insert.len()).unwrap_or(TextSize::from(u32::MAX));
            TextRange::at(current_indel.delete.start(), insert_len)
        };

        // figure out how much this Indel will shift future ranges from the initial source
        let offset_adjustment =
            u32::from(current_indel.delete.len()) as i32 - u32::from(new_range.len()) as i32;

        // insert any snippets before the text edit
        for (snippet_index, snippet_range) in snippets.peeking_take_while(|(_, range)| {
            offset_range(*range, source_text_offset).end() < new_range.start()
        }) {
            // adjust the snippet range into the corresponding initial source location
            let snippet_range = offset_range(snippet_range, source_text_offset);

            let snippet_range = if !stdx::always!(
                snippet_range.is_empty(),
                "placeholder range {:?} is before current text edit range {:?}",
                snippet_range,
                new_range
            ) {
                // only possible for tabstops, so make sure it's an empty/insert range
                TextRange::empty(snippet_range.start())
            } else {
                snippet_range
            };

            edits.push(snippet_text_edit(
                line_index,
                true,
                Indel { insert: format!("${snippet_index}"), delete: snippet_range },
            ))
        }

        if snippets.peek().is_some_and(|(_, range)| {
            new_range.intersect(offset_range(*range, source_text_offset)).is_some()
        }) {
            // at least one snippet edit intersects this text edit,
            // so gather all of the edits that intersect this text edit
            let mut all_snippets = snippets
                .peeking_take_while(|(_, range)| {
                    new_range.intersect(offset_range(*range, source_text_offset)).is_some()
                })
                .map(|(tabstop, range)| (tabstop, offset_range(range, source_text_offset)))
                .collect_vec();

            // ensure all of the ranges are wholly contained inside of the new range
            all_snippets.retain(|(_, range)| {
                    stdx::always!(
                        new_range.contains_range(*range),
                        "found placeholder range {:?} which wasn't fully inside of text edit's new range {:?}", range, new_range
                    )
                });

            let mut new_text = current_indel.insert;

            // find which snippet bits need to be escaped
            let escape_places =
                new_text.rmatch_indices(['\\', '$', '}']).map(|(insert, _)| insert).collect_vec();
            let mut escape_places = escape_places.into_iter().peekable();
            let mut escape_prior_bits = |new_text: &mut String, up_to: usize| {
                for before in escape_places.peeking_take_while(|insert| *insert >= up_to) {
                    new_text.insert(before, '\\');
                }
            };

            // insert snippets, and escaping any needed bits along the way
            for (index, range) in all_snippets.iter().rev() {
                let text_range = range - new_range.start();
                let (start, end) = (text_range.start().into(), text_range.end().into());

                if range.is_empty() {
                    escape_prior_bits(&mut new_text, start);
                    new_text.insert_str(start, &format!("${index}"));
                } else {
                    escape_prior_bits(&mut new_text, end);
                    new_text.insert(end, '}');
                    escape_prior_bits(&mut new_text, start);
                    new_text.insert_str(start, &format!("${{{index}:"));
                }
            }

            // escape any remaining bits
            escape_prior_bits(&mut new_text, 0);

            edits.push(snippet_text_edit(
                line_index,
                true,
                Indel { insert: new_text, delete: current_indel.delete },
            ))
        } else {
            // snippet edit was beyond the current one
            // since it wasn't consumed, it's available for the next pass
            edits.push(snippet_text_edit(line_index, false, current_indel));
        }

        // update the final source -> initial source mapping offset
        source_text_offset += offset_adjustment;
    }

    // insert any remaining tabstops
    edits.extend(snippets.map(|(snippet_index, snippet_range)| {
        // adjust the snippet range into the corresponding initial source location
        let snippet_range = offset_range(snippet_range, source_text_offset);

        let snippet_range = if !stdx::always!(
            snippet_range.is_empty(),
            "found placeholder snippet {:?} without a text edit",
            snippet_range
        ) {
            TextRange::empty(snippet_range.start())
        } else {
            snippet_range
        };

        snippet_text_edit(
            line_index,
            true,
            Indel { insert: format!("${snippet_index}"), delete: snippet_range },
        )
    }));

    edits
}

pub(crate) fn snippet_text_document_edit(
    snap: &GlobalStateSnapshot,
    is_snippet: bool,
    file_id: FileId,
    edit: TextEdit,
    snippet_edit: Option<SnippetEdit>,
) -> Cancellable<lsp_ext::SnippetTextDocumentEdit> {
    let text_document = optional_versioned_text_document_identifier(snap, file_id);
    let line_index = snap.file_line_index(file_id)?;
    let mut edits = if let Some(snippet_edit) = snippet_edit {
        merge_text_and_snippet_edits(&line_index, edit, snippet_edit)
    } else {
        edit.into_iter().map(|it| snippet_text_edit(&line_index, is_snippet, it)).collect()
    };

    if snap.analysis.is_library_file(file_id)? && snap.config.change_annotation_support() {
        for edit in &mut edits {
            edit.annotation_id = Some(outside_workspace_annotation_id())
        }
    }
    Ok(lsp_ext::SnippetTextDocumentEdit { text_document, edits })
}

pub(crate) fn snippet_text_document_ops(
    snap: &GlobalStateSnapshot,
    file_system_edit: FileSystemEdit,
) -> Cancellable<Vec<lsp_ext::SnippetDocumentChangeOperation>> {
    let mut ops = Vec::new();
    match file_system_edit {
        FileSystemEdit::CreateFile { dst, initial_contents } => {
            let uri = snap.anchored_path(&dst);
            let create_file = lsp_types::ResourceOp::Create(lsp_types::CreateFile {
                uri: uri.clone(),
                options: None,
                annotation_id: None,
            });
            ops.push(lsp_ext::SnippetDocumentChangeOperation::Op(create_file));
            if !initial_contents.is_empty() {
                let text_document =
                    lsp_types::OptionalVersionedTextDocumentIdentifier { uri, version: None };
                let text_edit = lsp_ext::SnippetTextEdit {
                    range: lsp_types::Range::default(),
                    new_text: initial_contents,
                    insert_text_format: Some(lsp_types::InsertTextFormat::PLAIN_TEXT),
                    annotation_id: None,
                };
                let edit_file =
                    lsp_ext::SnippetTextDocumentEdit { text_document, edits: vec![text_edit] };
                ops.push(lsp_ext::SnippetDocumentChangeOperation::Edit(edit_file));
            }
        }
        FileSystemEdit::MoveFile { src, dst } => {
            let old_uri = snap.file_id_to_url(src);
            let new_uri = snap.anchored_path(&dst);
            let mut rename_file =
                lsp_types::RenameFile { old_uri, new_uri, options: None, annotation_id: None };
            if snap.analysis.is_library_file(src).ok() == Some(true)
                && snap.config.change_annotation_support()
            {
                rename_file.annotation_id = Some(outside_workspace_annotation_id())
            }
            ops.push(lsp_ext::SnippetDocumentChangeOperation::Op(lsp_types::ResourceOp::Rename(
                rename_file,
            )))
        }
        FileSystemEdit::MoveDir { src, src_id, dst } => {
            let old_uri = snap.anchored_path(&src);
            let new_uri = snap.anchored_path(&dst);
            let mut rename_file =
                lsp_types::RenameFile { old_uri, new_uri, options: None, annotation_id: None };
            if snap.analysis.is_library_file(src_id).ok() == Some(true)
                && snap.config.change_annotation_support()
            {
                rename_file.annotation_id = Some(outside_workspace_annotation_id())
            }
            ops.push(lsp_ext::SnippetDocumentChangeOperation::Op(lsp_types::ResourceOp::Rename(
                rename_file,
            )))
        }
    }
    Ok(ops)
}

pub(crate) fn snippet_workspace_edit(
    snap: &GlobalStateSnapshot,
    mut source_change: SourceChange,
) -> Cancellable<lsp_ext::SnippetWorkspaceEdit> {
    let mut document_changes: Vec<lsp_ext::SnippetDocumentChangeOperation> = Vec::new();

    for op in &mut source_change.file_system_edits {
        if let FileSystemEdit::CreateFile { dst, initial_contents } = op {
            // replace with a placeholder to avoid cloneing the edit
            let op = FileSystemEdit::CreateFile {
                dst: dst.clone(),
                initial_contents: mem::take(initial_contents),
            };
            let ops = snippet_text_document_ops(snap, op)?;
            document_changes.extend_from_slice(&ops);
        }
    }
    for (file_id, (edit, snippet_edit)) in source_change.source_file_edits {
        let edit = snippet_text_document_edit(
            snap,
            source_change.is_snippet,
            file_id,
            edit,
            snippet_edit,
        )?;
        document_changes.push(lsp_ext::SnippetDocumentChangeOperation::Edit(edit));
    }
    for op in source_change.file_system_edits {
        if !matches!(op, FileSystemEdit::CreateFile { .. }) {
            let ops = snippet_text_document_ops(snap, op)?;
            document_changes.extend_from_slice(&ops);
        }
    }
    let mut workspace_edit = lsp_ext::SnippetWorkspaceEdit {
        changes: None,
        document_changes: Some(document_changes),
        change_annotations: None,
    };
    if snap.config.change_annotation_support() {
        workspace_edit.change_annotations = Some(
            once((
                outside_workspace_annotation_id(),
                lsp_types::ChangeAnnotation {
                    label: String::from("Edit outside of the workspace"),
                    needs_confirmation: Some(true),
                    description: Some(String::from(
                        "This edit lies outside of the workspace and may affect dependencies",
                    )),
                },
            ))
            .collect(),
        )
    }
    Ok(workspace_edit)
}

pub(crate) fn workspace_edit(
    snap: &GlobalStateSnapshot,
    source_change: SourceChange,
) -> Cancellable<lsp_types::WorkspaceEdit> {
    assert!(!source_change.is_snippet);
    snippet_workspace_edit(snap, source_change).map(|it| it.into())
}

impl From<lsp_ext::SnippetWorkspaceEdit> for lsp_types::WorkspaceEdit {
    fn from(snippet_workspace_edit: lsp_ext::SnippetWorkspaceEdit) -> lsp_types::WorkspaceEdit {
        lsp_types::WorkspaceEdit {
            changes: None,
            document_changes: snippet_workspace_edit.document_changes.map(|changes| {
                lsp_types::DocumentChanges::Operations(
                    changes
                        .into_iter()
                        .map(|change| match change {
                            lsp_ext::SnippetDocumentChangeOperation::Op(op) => {
                                lsp_types::DocumentChangeOperation::Op(op)
                            }
                            lsp_ext::SnippetDocumentChangeOperation::Edit(edit) => {
                                lsp_types::DocumentChangeOperation::Edit(
                                    lsp_types::TextDocumentEdit {
                                        text_document: edit.text_document,
                                        edits: edit.edits.into_iter().map(From::from).collect(),
                                    },
                                )
                            }
                        })
                        .collect(),
                )
            }),
            change_annotations: snippet_workspace_edit.change_annotations,
        }
    }
}

impl From<lsp_ext::SnippetTextEdit>
    for lsp_types::OneOf<lsp_types::TextEdit, lsp_types::AnnotatedTextEdit>
{
    fn from(
        lsp_ext::SnippetTextEdit { annotation_id, insert_text_format:_, new_text, range }: lsp_ext::SnippetTextEdit,
    ) -> Self {
        match annotation_id {
            Some(annotation_id) => lsp_types::OneOf::Right(lsp_types::AnnotatedTextEdit {
                text_edit: lsp_types::TextEdit { range, new_text },
                annotation_id,
            }),
            None => lsp_types::OneOf::Left(lsp_types::TextEdit { range, new_text }),
        }
    }
}

pub(crate) fn call_hierarchy_item(
    snap: &GlobalStateSnapshot,
    target: NavigationTarget,
) -> Cancellable<lsp_types::CallHierarchyItem> {
    let name = target.name.to_string();
    let detail = target.description.clone();
    let kind = target.kind.map(symbol_kind).unwrap_or(lsp_types::SymbolKind::FUNCTION);
    let (uri, range, selection_range) = location_info(snap, target)?;
    Ok(lsp_types::CallHierarchyItem {
        name,
        kind,
        tags: None,
        detail,
        uri,
        range,
        selection_range,
        data: None,
    })
}

pub(crate) fn code_action_kind(kind: AssistKind) -> lsp_types::CodeActionKind {
    match kind {
        AssistKind::None | AssistKind::Generate => lsp_types::CodeActionKind::EMPTY,
        AssistKind::QuickFix => lsp_types::CodeActionKind::QUICKFIX,
        AssistKind::Refactor => lsp_types::CodeActionKind::REFACTOR,
        AssistKind::RefactorExtract => lsp_types::CodeActionKind::REFACTOR_EXTRACT,
        AssistKind::RefactorInline => lsp_types::CodeActionKind::REFACTOR_INLINE,
        AssistKind::RefactorRewrite => lsp_types::CodeActionKind::REFACTOR_REWRITE,
    }
}

pub(crate) fn code_action(
    snap: &GlobalStateSnapshot,
    assist: Assist,
    resolve_data: Option<(usize, lsp_types::CodeActionParams)>,
) -> Cancellable<lsp_ext::CodeAction> {
    let mut res = lsp_ext::CodeAction {
        title: assist.label.to_string(),
        group: assist.group.filter(|_| snap.config.code_action_group()).map(|gr| gr.0),
        kind: Some(code_action_kind(assist.id.1)),
        edit: None,
        is_preferred: None,
        data: None,
        command: None,
    };

    if assist.trigger_signature_help && snap.config.client_commands().trigger_parameter_hints {
        res.command = Some(command::trigger_parameter_hints());
    }

    match (assist.source_change, resolve_data) {
        (Some(it), _) => res.edit = Some(snippet_workspace_edit(snap, it)?),
        (None, Some((index, code_action_params))) => {
            res.data = Some(lsp_ext::CodeActionData {
                id: format!("{}:{}:{index}", assist.id.0, assist.id.1.name()),
                code_action_params,
            });
        }
        (None, None) => {
            stdx::never!("assist should always be resolved if client can't do lazy resolving")
        }
    };
    Ok(res)
}

pub(crate) fn runnable(
    snap: &GlobalStateSnapshot,
    runnable: Runnable,
) -> Cancellable<lsp_ext::Runnable> {
    let config = snap.config.runnables();
    let spec = CargoTargetSpec::for_file(snap, runnable.nav.file_id)?;
    let workspace_root = spec.as_ref().map(|it| it.workspace_root.clone());
    let target = spec.as_ref().map(|s| s.target.clone());
    let (cargo_args, executable_args) =
        CargoTargetSpec::runnable_args(snap, spec, &runnable.kind, &runnable.cfg);
    let label = runnable.label(target);
    let location = location_link(snap, None, runnable.nav)?;

    Ok(lsp_ext::Runnable {
        label,
        location: Some(location),
        kind: lsp_ext::RunnableKind::Cargo,
        args: lsp_ext::CargoRunnable {
            workspace_root: workspace_root.map(|it| it.into()),
            override_cargo: config.override_cargo,
            cargo_args,
            cargo_extra_args: config.cargo_extra_args,
            executable_args,
            expect_test: None,
        },
    })
}

pub(crate) fn code_lens(
    acc: &mut Vec<lsp_types::CodeLens>,
    snap: &GlobalStateSnapshot,
    annotation: Annotation,
) -> Cancellable<()> {
    let client_commands_config = snap.config.client_commands();
    match annotation.kind {
        AnnotationKind::Runnable(run) => {
            let line_index = snap.file_line_index(run.nav.file_id)?;
            let annotation_range = range(&line_index, annotation.range);

            let title = run.title();
            let can_debug = match run.kind {
                ide::RunnableKind::DocTest { .. } => false,
                ide::RunnableKind::TestMod { .. }
                | ide::RunnableKind::Test { .. }
                | ide::RunnableKind::Bench { .. }
                | ide::RunnableKind::Bin => true,
            };
            let r = runnable(snap, run)?;

            let lens_config = snap.config.lens();
            if lens_config.run
                && client_commands_config.run_single
                && r.args.workspace_root.is_some()
            {
                let command = command::run_single(&r, &title);
                acc.push(lsp_types::CodeLens {
                    range: annotation_range,
                    command: Some(command),
                    data: None,
                })
            }
            if lens_config.debug && can_debug && client_commands_config.debug_single {
                let command = command::debug_single(&r);
                acc.push(lsp_types::CodeLens {
                    range: annotation_range,
                    command: Some(command),
                    data: None,
                })
            }
            if lens_config.interpret {
                let command = command::interpret_single(&r);
                acc.push(lsp_types::CodeLens {
                    range: annotation_range,
                    command: Some(command),
                    data: None,
                })
            }
        }
        AnnotationKind::HasImpls { pos, data } => {
            if !client_commands_config.show_reference {
                return Ok(());
            }
            let line_index = snap.file_line_index(pos.file_id)?;
            let annotation_range = range(&line_index, annotation.range);
            let url = url(snap, pos.file_id);
            let pos = position(&line_index, pos.offset);

            let id = lsp_types::TextDocumentIdentifier { uri: url.clone() };

            let doc_pos = lsp_types::TextDocumentPositionParams::new(id, pos);

            let goto_params = lsp_types::request::GotoImplementationParams {
                text_document_position_params: doc_pos,
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            };

            let command = data.map(|ranges| {
                let locations: Vec<lsp_types::Location> = ranges
                    .into_iter()
                    .filter_map(|target| {
                        location(
                            snap,
                            FileRange { file_id: target.file_id, range: target.full_range },
                        )
                        .ok()
                    })
                    .collect();

                command::show_references(
                    implementation_title(locations.len()),
                    &url,
                    pos,
                    locations,
                )
            });

            acc.push(lsp_types::CodeLens {
                range: annotation_range,
                command,
                data: (|| {
                    let version = snap.url_file_version(&url)?;
                    Some(
                        to_value(lsp_ext::CodeLensResolveData {
                            version,
                            kind: lsp_ext::CodeLensResolveDataKind::Impls(goto_params),
                        })
                        .unwrap(),
                    )
                })(),
            })
        }
        AnnotationKind::HasReferences { pos, data } => {
            if !client_commands_config.show_reference {
                return Ok(());
            }
            let line_index = snap.file_line_index(pos.file_id)?;
            let annotation_range = range(&line_index, annotation.range);
            let url = url(snap, pos.file_id);
            let pos = position(&line_index, pos.offset);

            let id = lsp_types::TextDocumentIdentifier { uri: url.clone() };

            let doc_pos = lsp_types::TextDocumentPositionParams::new(id, pos);

            let command = data.map(|ranges| {
                let locations: Vec<lsp_types::Location> =
                    ranges.into_iter().filter_map(|range| location(snap, range).ok()).collect();

                command::show_references(reference_title(locations.len()), &url, pos, locations)
            });

            acc.push(lsp_types::CodeLens {
                range: annotation_range,
                command,
                data: (|| {
                    let version = snap.url_file_version(&url)?;
                    Some(
                        to_value(lsp_ext::CodeLensResolveData {
                            version,
                            kind: lsp_ext::CodeLensResolveDataKind::References(doc_pos),
                        })
                        .unwrap(),
                    )
                })(),
            })
        }
    }
    Ok(())
}

pub(crate) fn test_item(
    snap: &GlobalStateSnapshot,
    test_item: ide::TestItem,
    line_index: Option<&LineIndex>,
) -> lsp_ext::TestItem {
    lsp_ext::TestItem {
        id: test_item.id,
        label: test_item.label,
        kind: match test_item.kind {
            ide::TestItemKind::Crate(id) => 'b: {
                let Some((cargo_ws, target)) = snap.cargo_target_for_crate_root(id) else {
                    break 'b lsp_ext::TestItemKind::Package;
                };
                let target = &cargo_ws[target];
                match target.kind {
                    project_model::TargetKind::Bin
                    | project_model::TargetKind::Lib { .. }
                    | project_model::TargetKind::Example
                    | project_model::TargetKind::BuildScript
                    | project_model::TargetKind::Other => lsp_ext::TestItemKind::Package,
                    project_model::TargetKind::Test | project_model::TargetKind::Bench => {
                        lsp_ext::TestItemKind::Test
                    }
                }
            }
            ide::TestItemKind::Module => lsp_ext::TestItemKind::Module,
            ide::TestItemKind::Function => lsp_ext::TestItemKind::Test,
        },
        can_resolve_children: matches!(
            test_item.kind,
            ide::TestItemKind::Crate(_) | ide::TestItemKind::Module
        ),
        parent: test_item.parent,
        text_document: test_item
            .file
            .map(|f| lsp_types::TextDocumentIdentifier { uri: url(snap, f) }),
        range: line_index.and_then(|l| Some(range(l, test_item.text_range?))),
        runnable: test_item.runnable.and_then(|r| runnable(snap, r).ok()),
    }
}

pub(crate) mod command {
    use ide::{FileRange, NavigationTarget};
    use serde_json::to_value;

    use crate::{
        global_state::GlobalStateSnapshot,
        lsp::to_proto::{location, location_link},
        lsp_ext,
    };

    pub(crate) fn show_references(
        title: String,
        uri: &lsp_types::Url,
        position: lsp_types::Position,
        locations: Vec<lsp_types::Location>,
    ) -> lsp_types::Command {
        // We cannot use the 'editor.action.showReferences' command directly
        // because that command requires vscode types which we convert in the handler
        // on the client side.

        lsp_types::Command {
            title,
            command: "rust-analyzer.showReferences".into(),
            arguments: Some(vec![
                to_value(uri).unwrap(),
                to_value(position).unwrap(),
                to_value(locations).unwrap(),
            ]),
        }
    }

    pub(crate) fn run_single(runnable: &lsp_ext::Runnable, title: &str) -> lsp_types::Command {
        lsp_types::Command {
            title: title.to_owned(),
            command: "rust-analyzer.runSingle".into(),
            arguments: Some(vec![to_value(runnable).unwrap()]),
        }
    }

    pub(crate) fn debug_single(runnable: &lsp_ext::Runnable) -> lsp_types::Command {
        lsp_types::Command {
            title: "Debug".into(),
            command: "rust-analyzer.debugSingle".into(),
            arguments: Some(vec![to_value(runnable).unwrap()]),
        }
    }

    pub(crate) fn interpret_single(_runnable: &lsp_ext::Runnable) -> lsp_types::Command {
        lsp_types::Command {
            title: "Interpret".into(),
            command: "rust-analyzer.interpretFunction".into(),
            // FIXME: use the `_runnable` here.
            arguments: Some(vec![]),
        }
    }

    pub(crate) fn goto_location(
        snap: &GlobalStateSnapshot,
        nav: &NavigationTarget,
    ) -> Option<lsp_types::Command> {
        let value = if snap.config.location_link() {
            let link = location_link(snap, None, nav.clone()).ok()?;
            to_value(link).ok()?
        } else {
            let range = FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() };
            let location = location(snap, range).ok()?;
            to_value(location).ok()?
        };

        Some(lsp_types::Command {
            title: nav.name.to_string(),
            command: "rust-analyzer.gotoLocation".into(),
            arguments: Some(vec![value]),
        })
    }

    pub(crate) fn trigger_parameter_hints() -> lsp_types::Command {
        lsp_types::Command {
            title: "triggerParameterHints".into(),
            command: "rust-analyzer.triggerParameterHints".into(),
            arguments: None,
        }
    }
}

pub(crate) fn implementation_title(count: usize) -> String {
    if count == 1 {
        "1 implementation".into()
    } else {
        format!("{count} implementations")
    }
}

pub(crate) fn reference_title(count: usize) -> String {
    if count == 1 {
        "1 reference".into()
    } else {
        format!("{count} references")
    }
}

pub(crate) fn markup_content(
    markup: Markup,
    kind: ide::HoverDocFormat,
) -> lsp_types::MarkupContent {
    let kind = match kind {
        ide::HoverDocFormat::Markdown => lsp_types::MarkupKind::Markdown,
        ide::HoverDocFormat::PlainText => lsp_types::MarkupKind::PlainText,
    };
    let value = format_docs(&Documentation::new(markup.into()));
    lsp_types::MarkupContent { kind, value }
}

pub(crate) fn rename_error(err: RenameError) -> LspError {
    // This is wrong, but we don't have a better alternative I suppose?
    // https://github.com/microsoft/language-server-protocol/issues/1341
    invalid_params_error(err.to_string())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide::{Analysis, FilePosition};
    use ide_db::source_change::Snippet;
    use test_utils::extract_offset;
    use triomphe::Arc;

    use super::*;

    #[test]
    fn conv_fold_line_folding_only_fixup() {
        let text = r#"mod a;
mod b;
mod c;

fn main() {
    if cond {
        a::do_a();
    } else {
        b::do_b();
    }
}"#;

        let (analysis, file_id) = Analysis::from_single_file(text.to_owned());
        let folds = analysis.folding_ranges(file_id).unwrap();
        assert_eq!(folds.len(), 4);

        let line_index = LineIndex {
            index: Arc::new(ide::LineIndex::new(text)),
            endings: LineEndings::Unix,
            encoding: PositionEncoding::Utf8,
        };
        let converted: Vec<lsp_types::FoldingRange> =
            folds.into_iter().map(|it| folding_range(text, &line_index, true, it)).collect();

        let expected_lines = [(0, 2), (4, 10), (5, 6), (7, 9)];
        assert_eq!(converted.len(), expected_lines.len());
        for (folding_range, (start_line, end_line)) in converted.iter().zip(expected_lines.iter()) {
            assert_eq!(folding_range.start_line, *start_line);
            assert_eq!(folding_range.start_character, None);
            assert_eq!(folding_range.end_line, *end_line);
            assert_eq!(folding_range.end_character, None);
        }
    }

    #[test]
    fn calling_function_with_ignored_code_in_signature() {
        let text = r#"
fn foo() {
    bar($0);
}
/// ```
/// # use crate::bar;
/// bar(5);
/// ```
fn bar(_: usize) {}
"#;

        let (offset, text) = extract_offset(text);
        let (analysis, file_id) = Analysis::from_single_file(text);
        let help = signature_help(
            analysis.signature_help(FilePosition { file_id, offset }).unwrap().unwrap(),
            CallInfoConfig { params_only: false, docs: true },
            false,
        );
        let docs = match &help.signatures[help.active_signature.unwrap() as usize].documentation {
            Some(lsp_types::Documentation::MarkupContent(content)) => &content.value,
            _ => panic!("documentation contains markup"),
        };
        assert!(docs.contains("bar(5)"));
        assert!(!docs.contains("use crate::bar"));
    }

    #[track_caller]
    fn check_rendered_snippets(edit: TextEdit, snippets: SnippetEdit, expect: Expect) {
        check_rendered_snippets_in_source(
            r"/* place to put all ranges in */",
            edit,
            snippets,
            expect,
        );
    }

    #[track_caller]
    fn check_rendered_snippets_in_source(
        ra_fixture: &str,
        edit: TextEdit,
        snippets: SnippetEdit,
        expect: Expect,
    ) {
        let source = stdx::trim_indent(ra_fixture);
        let endings = if source.contains('\r') { LineEndings::Dos } else { LineEndings::Unix };
        let line_index = LineIndex {
            index: Arc::new(ide::LineIndex::new(&source)),
            endings,
            encoding: PositionEncoding::Utf8,
        };

        let res = merge_text_and_snippet_edits(&line_index, edit, snippets);

        // Ensure that none of the ranges overlap
        {
            let mut sorted = res.clone();
            sorted.sort_by_key(|edit| (edit.range.start, edit.range.end));
            let disjoint_ranges = sorted
                .iter()
                .zip(sorted.iter().skip(1))
                .all(|(l, r)| l.range.end <= r.range.start || l == r);
            assert!(disjoint_ranges, "ranges overlap for {res:#?}");
        }

        expect.assert_debug_eq(&res);
    }

    #[test]
    fn snippet_rendering_only_tabstops() {
        let edit = TextEdit::builder().finish();
        let snippets = SnippetEdit::new(vec![
            Snippet::Tabstop(0.into()),
            Snippet::Tabstop(0.into()),
            Snippet::Tabstop(1.into()),
            Snippet::Tabstop(1.into()),
        ]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "$1",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "$2",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 1,
                        },
                        end: Position {
                            line: 0,
                            character: 1,
                        },
                    },
                    new_text: "$3",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 1,
                        },
                        end: Position {
                            line: 0,
                            character: 1,
                        },
                    },
                    new_text: "$0",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
            ]
        "#]],
        );
    }

    #[test]
    fn snippet_rendering_only_text_edits() {
        let mut edit = TextEdit::builder();
        edit.insert(0.into(), "abc".to_owned());
        edit.insert(3.into(), "def".to_owned());
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "abc",
                    insert_text_format: None,
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 3,
                        },
                        end: Position {
                            line: 0,
                            character: 3,
                        },
                    },
                    new_text: "def",
                    insert_text_format: None,
                    annotation_id: None,
                },
            ]
        "#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstop_after_text_edit() {
        let mut edit = TextEdit::builder();
        edit.insert(0.into(), "abc".to_owned());
        let edit = edit.finish();
        // Note: tabstops are positioned in the source where all text edits have been applied
        let snippets = SnippetEdit::new(vec![Snippet::Tabstop(10.into())]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "abc",
                    insert_text_format: None,
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 7,
                        },
                        end: Position {
                            line: 0,
                            character: 7,
                        },
                    },
                    new_text: "$0",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
            ]
        "#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstops_before_text_edit() {
        let mut edit = TextEdit::builder();
        edit.insert(2.into(), "abc".to_owned());
        let edit = edit.finish();
        let snippets =
            SnippetEdit::new(vec![Snippet::Tabstop(0.into()), Snippet::Tabstop(0.into())]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
                [
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 0,
                            },
                            end: Position {
                                line: 0,
                                character: 0,
                            },
                        },
                        new_text: "$1",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 0,
                            },
                            end: Position {
                                line: 0,
                                character: 0,
                            },
                        },
                        new_text: "$0",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 2,
                            },
                            end: Position {
                                line: 0,
                                character: 2,
                            },
                        },
                        new_text: "abc",
                        insert_text_format: None,
                        annotation_id: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstops_between_text_edits() {
        let mut edit = TextEdit::builder();
        edit.insert(0.into(), "abc".to_owned());
        edit.insert(7.into(), "abc".to_owned());
        let edit = edit.finish();
        // Note: tabstops are positioned in the source where all text edits have been applied
        let snippets =
            SnippetEdit::new(vec![Snippet::Tabstop(7.into()), Snippet::Tabstop(7.into())]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "abc",
                    insert_text_format: None,
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 4,
                        },
                        end: Position {
                            line: 0,
                            character: 4,
                        },
                    },
                    new_text: "$1",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 4,
                        },
                        end: Position {
                            line: 0,
                            character: 4,
                        },
                    },
                    new_text: "$0",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 7,
                        },
                        end: Position {
                            line: 0,
                            character: 7,
                        },
                    },
                    new_text: "abc",
                    insert_text_format: None,
                    annotation_id: None,
                },
            ]
        "#]],
        );
    }

    #[test]
    fn snippet_rendering_multiple_tabstops_in_text_edit() {
        let mut edit = TextEdit::builder();
        edit.insert(0.into(), "abcdefghijkl".to_owned());
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![
            Snippet::Tabstop(0.into()),
            Snippet::Tabstop(5.into()),
            Snippet::Tabstop(12.into()),
        ]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "$1abcde$2fghijkl$0",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
            ]
        "#]],
        );
    }

    #[test]
    fn snippet_rendering_multiple_placeholders_in_text_edit() {
        let mut edit = TextEdit::builder();
        edit.insert(0.into(), "abcdefghijkl".to_owned());
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![
            Snippet::Placeholder(TextRange::new(0.into(), 3.into())),
            Snippet::Placeholder(TextRange::new(5.into(), 7.into())),
            Snippet::Placeholder(TextRange::new(10.into(), 12.into())),
        ]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: 0,
                        },
                        end: Position {
                            line: 0,
                            character: 0,
                        },
                    },
                    new_text: "${1:abc}de${2:fg}hij${0:kl}",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
            ]
        "#]],
        );
    }

    #[test]
    fn snippet_rendering_escape_snippet_bits() {
        // only needed for snippet formats
        let mut edit = TextEdit::builder();
        edit.insert(0.into(), r"$ab{}$c\def".to_owned());
        edit.insert(8.into(), r"ghi\jk<-check_insert_here$".to_owned());
        edit.insert(10.into(), r"a\\b\\c{}$".to_owned());
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![
            Snippet::Placeholder(TextRange::new(1.into(), 9.into())),
            Snippet::Tabstop(25.into()),
        ]);

        check_rendered_snippets(
            edit,
            snippets,
            expect![[r#"
                [
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 0,
                            },
                            end: Position {
                                line: 0,
                                character: 0,
                            },
                        },
                        new_text: "\\$${1:ab{\\}\\$c\\\\d}ef",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 8,
                            },
                            end: Position {
                                line: 0,
                                character: 8,
                            },
                        },
                        new_text: "ghi\\\\jk$0<-check_insert_here\\$",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: 10,
                            },
                            end: Position {
                                line: 0,
                                character: 10,
                            },
                        },
                        new_text: "a\\\\b\\\\c{}$",
                        insert_text_format: None,
                        annotation_id: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstop_adjust_offset_deleted() {
        // negative offset from inserting a smaller range
        let mut edit = TextEdit::builder();
        edit.replace(TextRange::new(47.into(), 56.into()), "let".to_owned());
        edit.replace(
            TextRange::new(57.into(), 89.into()),
            "disabled = false;\n    ProcMacro {\n        disabled,\n    }".to_owned(),
        );
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![Snippet::Tabstop(51.into())]);

        check_rendered_snippets_in_source(
            r"
fn expander_to_proc_macro() -> ProcMacro {
    ProcMacro {
        disabled: false,
    }
}

struct ProcMacro {
    disabled: bool,
}",
            edit,
            snippets,
            expect![[r#"
                [
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 4,
                            },
                            end: Position {
                                line: 1,
                                character: 13,
                            },
                        },
                        new_text: "let",
                        insert_text_format: None,
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 14,
                            },
                            end: Position {
                                line: 3,
                                character: 5,
                            },
                        },
                        new_text: "$0disabled = false;\n    ProcMacro {\n        disabled,\n    \\}",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstop_adjust_offset_added() {
        // positive offset from inserting a larger range
        let mut edit = TextEdit::builder();
        edit.replace(TextRange::new(39.into(), 40.into()), "let".to_owned());
        edit.replace(
            TextRange::new(41.into(), 73.into()),
            "disabled = false;\n    ProcMacro {\n        disabled,\n    }".to_owned(),
        );
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![Snippet::Tabstop(43.into())]);

        check_rendered_snippets_in_source(
            r"
fn expander_to_proc_macro() -> P {
    P {
        disabled: false,
    }
}

struct P {
    disabled: bool,
}",
            edit,
            snippets,
            expect![[r#"
                [
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 4,
                            },
                            end: Position {
                                line: 1,
                                character: 5,
                            },
                        },
                        new_text: "let",
                        insert_text_format: None,
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 6,
                            },
                            end: Position {
                                line: 3,
                                character: 5,
                            },
                        },
                        new_text: "$0disabled = false;\n    ProcMacro {\n        disabled,\n    \\}",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn snippet_rendering_placeholder_adjust_offset_deleted() {
        // negative offset from inserting a smaller range
        let mut edit = TextEdit::builder();
        edit.replace(TextRange::new(47.into(), 56.into()), "let".to_owned());
        edit.replace(
            TextRange::new(57.into(), 89.into()),
            "disabled = false;\n    ProcMacro {\n        disabled,\n    }".to_owned(),
        );
        let edit = edit.finish();
        let snippets =
            SnippetEdit::new(vec![Snippet::Placeholder(TextRange::new(51.into(), 59.into()))]);

        check_rendered_snippets_in_source(
            r"
fn expander_to_proc_macro() -> ProcMacro {
    ProcMacro {
        disabled: false,
    }
}

struct ProcMacro {
    disabled: bool,
}",
            edit,
            snippets,
            expect![[r#"
                [
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 4,
                            },
                            end: Position {
                                line: 1,
                                character: 13,
                            },
                        },
                        new_text: "let",
                        insert_text_format: None,
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 14,
                            },
                            end: Position {
                                line: 3,
                                character: 5,
                            },
                        },
                        new_text: "${0:disabled} = false;\n    ProcMacro {\n        disabled,\n    \\}",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn snippet_rendering_placeholder_adjust_offset_added() {
        // positive offset from inserting a larger range
        let mut edit = TextEdit::builder();
        edit.replace(TextRange::new(39.into(), 40.into()), "let".to_owned());
        edit.replace(
            TextRange::new(41.into(), 73.into()),
            "disabled = false;\n    ProcMacro {\n        disabled,\n    }".to_owned(),
        );
        let edit = edit.finish();
        let snippets =
            SnippetEdit::new(vec![Snippet::Placeholder(TextRange::new(43.into(), 51.into()))]);

        check_rendered_snippets_in_source(
            r"
fn expander_to_proc_macro() -> P {
    P {
        disabled: false,
    }
}

struct P {
    disabled: bool,
}",
            edit,
            snippets,
            expect![[r#"
                [
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 4,
                            },
                            end: Position {
                                line: 1,
                                character: 5,
                            },
                        },
                        new_text: "let",
                        insert_text_format: None,
                        annotation_id: None,
                    },
                    SnippetTextEdit {
                        range: Range {
                            start: Position {
                                line: 1,
                                character: 6,
                            },
                            end: Position {
                                line: 3,
                                character: 5,
                            },
                        },
                        new_text: "${0:disabled} = false;\n    ProcMacro {\n        disabled,\n    \\}",
                        insert_text_format: Some(
                            Snippet,
                        ),
                        annotation_id: None,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstop_adjust_offset_between_text_edits() {
        // inserting between edits, tabstop should be at (1, 14)
        let mut edit = TextEdit::builder();
        edit.replace(TextRange::new(47.into(), 56.into()), "let".to_owned());
        edit.replace(
            TextRange::new(58.into(), 90.into()),
            "disabled = false;\n    ProcMacro {\n        disabled,\n    }".to_owned(),
        );
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![Snippet::Tabstop(51.into())]);

        // add an extra space between `ProcMacro` and `{` to insert the tabstop at
        check_rendered_snippets_in_source(
            r"
fn expander_to_proc_macro() -> ProcMacro {
    ProcMacro  {
        disabled: false,
    }
}

struct ProcMacro {
    disabled: bool,
}",
            edit,
            snippets,
            expect![[r#"
    [
        SnippetTextEdit {
            range: Range {
                start: Position {
                    line: 1,
                    character: 4,
                },
                end: Position {
                    line: 1,
                    character: 13,
                },
            },
            new_text: "let",
            insert_text_format: None,
            annotation_id: None,
        },
        SnippetTextEdit {
            range: Range {
                start: Position {
                    line: 1,
                    character: 14,
                },
                end: Position {
                    line: 1,
                    character: 14,
                },
            },
            new_text: "$0",
            insert_text_format: Some(
                Snippet,
            ),
            annotation_id: None,
        },
        SnippetTextEdit {
            range: Range {
                start: Position {
                    line: 1,
                    character: 15,
                },
                end: Position {
                    line: 3,
                    character: 5,
                },
            },
            new_text: "disabled = false;\n    ProcMacro {\n        disabled,\n    }",
            insert_text_format: None,
            annotation_id: None,
        },
    ]
"#]],
        );
    }

    #[test]
    fn snippet_rendering_tabstop_adjust_offset_after_text_edits() {
        // inserting after edits, tabstop should be before the closing curly of the fn
        let mut edit = TextEdit::builder();
        edit.replace(TextRange::new(47.into(), 56.into()), "let".to_owned());
        edit.replace(
            TextRange::new(57.into(), 89.into()),
            "disabled = false;\n    ProcMacro {\n        disabled,\n    }".to_owned(),
        );
        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![Snippet::Tabstop(109.into())]);

        check_rendered_snippets_in_source(
            r"
fn expander_to_proc_macro() -> ProcMacro {
    ProcMacro {
        disabled: false,
    }
}

struct ProcMacro {
    disabled: bool,
}",
            edit,
            snippets,
            expect![[r#"
    [
        SnippetTextEdit {
            range: Range {
                start: Position {
                    line: 1,
                    character: 4,
                },
                end: Position {
                    line: 1,
                    character: 13,
                },
            },
            new_text: "let",
            insert_text_format: None,
            annotation_id: None,
        },
        SnippetTextEdit {
            range: Range {
                start: Position {
                    line: 1,
                    character: 14,
                },
                end: Position {
                    line: 3,
                    character: 5,
                },
            },
            new_text: "disabled = false;\n    ProcMacro {\n        disabled,\n    }",
            insert_text_format: None,
            annotation_id: None,
        },
        SnippetTextEdit {
            range: Range {
                start: Position {
                    line: 4,
                    character: 0,
                },
                end: Position {
                    line: 4,
                    character: 0,
                },
            },
            new_text: "$0",
            insert_text_format: Some(
                Snippet,
            ),
            annotation_id: None,
        },
    ]
"#]],
        );
    }

    #[test]
    fn snippet_rendering_handle_dos_line_endings() {
        // unix -> dos conversion should be handled after placing snippets
        let mut edit = TextEdit::builder();
        edit.insert(6.into(), "\n\n->".to_owned());

        let edit = edit.finish();
        let snippets = SnippetEdit::new(vec![Snippet::Tabstop(10.into())]);

        check_rendered_snippets_in_source(
            "yeah\r\n<-tabstop here",
            edit,
            snippets,
            expect![[r#"
            [
                SnippetTextEdit {
                    range: Range {
                        start: Position {
                            line: 1,
                            character: 0,
                        },
                        end: Position {
                            line: 1,
                            character: 0,
                        },
                    },
                    new_text: "\r\n\r\n->$0",
                    insert_text_format: Some(
                        Snippet,
                    ),
                    annotation_id: None,
                },
            ]
        "#]],
        )
    }

    // `Url` is not able to parse windows paths on unix machines.
    #[test]
    #[cfg(target_os = "windows")]
    fn test_lowercase_drive_letter() {
        use std::path::Path;

        let url = url_from_abs_path(Path::new("C:\\Test").try_into().unwrap());
        assert_eq!(url.to_string(), "file:///c:/Test");

        let url = url_from_abs_path(Path::new(r#"\\localhost\C$\my_dir"#).try_into().unwrap());
        assert_eq!(url.to_string(), "file://localhost/C$/my_dir");
    }
}
