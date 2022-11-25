//! Conversion of rust-analyzer specific types to lsp_types equivalents.
use std::{
    iter::once,
    path,
    sync::atomic::{AtomicU32, Ordering},
};

use ide::{
    Annotation, AnnotationKind, Assist, AssistKind, Cancellable, CompletionItem,
    CompletionItemKind, CompletionRelevance, Documentation, FileId, FileRange, FileSystemEdit,
    Fold, FoldKind, Highlight, HlMod, HlOperator, HlPunct, HlRange, HlTag, Indel, InlayHint,
    InlayHintLabel, InlayKind, Markup, NavigationTarget, ReferenceCategory, RenameError, Runnable,
    Severity, SignatureHelp, SourceChange, StructureNodeKind, SymbolKind, TextEdit, TextRange,
    TextSize,
};
use itertools::Itertools;
use serde_json::to_value;
use vfs::AbsPath;

use crate::{
    cargo_target_spec::CargoTargetSpec,
    config::{CallInfoConfig, Config},
    global_state::GlobalStateSnapshot,
    line_index::{LineEndings, LineIndex, OffsetEncoding},
    lsp_ext,
    lsp_utils::invalid_params_error,
    semantic_tokens, Result,
};

pub(crate) fn position(line_index: &LineIndex, offset: TextSize) -> lsp_types::Position {
    let line_col = line_index.index.line_col(offset);
    match line_index.encoding {
        OffsetEncoding::Utf8 => lsp_types::Position::new(line_col.line, line_col.col),
        OffsetEncoding::Utf16 => {
            let line_col = line_index.index.to_utf16(line_col);
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
        SymbolKind::Trait => lsp_types::SymbolKind::INTERFACE,
        SymbolKind::Macro
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
) -> lsp_types::DocumentHighlightKind {
    match category {
        ReferenceCategory::Read => lsp_types::DocumentHighlightKind::READ,
        ReferenceCategory::Write => lsp_types::DocumentHighlightKind::WRITE,
    }
}

pub(crate) fn diagnostic_severity(severity: Severity) -> lsp_types::DiagnosticSeverity {
    match severity {
        Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
        Severity::WeakWarning => lsp_types::DiagnosticSeverity::HINT,
    }
}

pub(crate) fn documentation(documentation: Documentation) -> lsp_types::Documentation {
    let value = crate::markdown::format_docs(documentation.as_str());
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
            SymbolKind::Module => lsp_types::CompletionItemKind::MODULE,
            SymbolKind::SelfParam => lsp_types::CompletionItemKind::VALUE,
            SymbolKind::SelfType => lsp_types::CompletionItemKind::TYPE_PARAMETER,
            SymbolKind::Static => lsp_types::CompletionItemKind::VALUE,
            SymbolKind::Struct => lsp_types::CompletionItemKind::STRUCT,
            SymbolKind::Trait => lsp_types::CompletionItemKind::INTERFACE,
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
    let max_relevance = items.iter().map(|it| it.relevance().score()).max().unwrap_or_default();
    let mut res = Vec::with_capacity(items.len());
    for item in items {
        completion_item(&mut res, config, line_index, &tdpp, max_relevance, item)
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
    let insert_replace_support = config.insert_replace_support().then(|| tdpp.position);
    let mut additional_text_edits = Vec::new();

    // LSP does not allow arbitrary edits in completion, so we have to do a
    // non-trivial mapping here.
    let text_edit = {
        let mut text_edit = None;
        let source_range = item.source_range();
        for indel in item.text_edit().iter() {
            if indel.delete.contains_range(source_range) {
                text_edit = Some(if indel.delete == source_range {
                    self::completion_text_edit(line_index, insert_replace_support, indel.clone())
                } else {
                    assert!(source_range.end() == indel.delete.end());
                    let range1 = TextRange::new(indel.delete.start(), source_range.start());
                    let range2 = source_range;
                    let indel1 = Indel::replace(range1, String::new());
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

    let insert_text_format = item.is_snippet().then(|| lsp_types::InsertTextFormat::SNIPPET);
    let tags = item.deprecated().then(|| vec![lsp_types::CompletionItemTag::DEPRECATED]);
    let command = if item.trigger_call_info() && config.client_commands().trigger_parameter_hints {
        Some(command::trigger_parameter_hints())
    } else {
        None
    };

    let mut lsp_item = lsp_types::CompletionItem {
        label: item.label().to_string(),
        detail: item.detail().map(|it| it.to_string()),
        filter_text: Some(item.lookup().to_string()),
        kind: Some(completion_item_kind(item.kind())),
        text_edit: Some(text_edit),
        additional_text_edits: Some(additional_text_edits),
        documentation: item.documentation().map(documentation),
        deprecated: Some(item.deprecated()),
        tags,
        command,
        insert_text_format,
        ..Default::default()
    };

    if config.completion_label_details_support() {
        lsp_item.label_details = Some(lsp_types::CompletionItemLabelDetails {
            detail: None,
            description: lsp_item.detail.clone(),
        });
    }

    set_score(&mut lsp_item, max_relevance, item.relevance());

    if config.completion().enable_imports_on_the_fly {
        if let imports @ [_, ..] = item.imports_to_add() {
            let imports: Vec<_> = imports
                .iter()
                .filter_map(|import_edit| {
                    let import_path = &import_edit.import_path;
                    let import_name = import_path.segments().last()?;
                    Some(lsp_ext::CompletionImport {
                        full_import_path: import_path.to_string(),
                        imported_name: import_name.to_string(),
                    })
                })
                .collect();
            if !imports.is_empty() {
                let data = lsp_ext::CompletionResolveData { position: tdpp.clone(), imports };
                lsp_item.data = Some(to_value(data).unwrap());
            }
        }
    }

    if let Some((mutability, offset, relevance)) = item.ref_match() {
        let mut lsp_item_with_ref = lsp_item.clone();
        set_score(&mut lsp_item_with_ref, max_relevance, relevance);
        lsp_item_with_ref.label =
            format!("&{}{}", mutability.as_keyword_for_ref(), lsp_item_with_ref.label);
        lsp_item_with_ref.additional_text_edits.get_or_insert_with(Default::default).push(
            self::text_edit(
                line_index,
                Indel::insert(offset, format!("&{}", mutability.as_keyword_for_ref())),
            ),
        );

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
        res.sort_text = Some(format!("{:08x}", sort_score));
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
                    label: lsp_types::ParameterLabel::Simple(label.to_string()),
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
            value: doc,
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
    line_index: &LineIndex,
    render_colons: bool,
    mut inlay_hint: InlayHint,
) -> Result<lsp_types::InlayHint> {
    match inlay_hint.kind {
        InlayKind::ParameterHint if render_colons => inlay_hint.label.append_str(":"),
        InlayKind::TypeHint if render_colons => inlay_hint.label.prepend_str(": "),
        InlayKind::ClosureReturnTypeHint => inlay_hint.label.prepend_str(" -> "),
        _ => {}
    }

    Ok(lsp_types::InlayHint {
        position: match inlay_hint.kind {
            // before annotated thing
            InlayKind::ParameterHint
            | InlayKind::ImplicitReborrowHint
            | InlayKind::BindingModeHint => position(line_index, inlay_hint.range.start()),
            // after annotated thing
            InlayKind::ClosureReturnTypeHint
            | InlayKind::TypeHint
            | InlayKind::ChainingHint
            | InlayKind::GenericParamListHint
            | InlayKind::LifetimeHint
            | InlayKind::ClosingBraceHint => position(line_index, inlay_hint.range.end()),
        },
        padding_left: Some(match inlay_hint.kind {
            InlayKind::TypeHint => !render_colons,
            InlayKind::ChainingHint | InlayKind::ClosingBraceHint => true,
            InlayKind::BindingModeHint
            | InlayKind::ClosureReturnTypeHint
            | InlayKind::GenericParamListHint
            | InlayKind::ImplicitReborrowHint
            | InlayKind::LifetimeHint
            | InlayKind::ParameterHint => false,
        }),
        padding_right: Some(match inlay_hint.kind {
            InlayKind::ChainingHint
            | InlayKind::ClosureReturnTypeHint
            | InlayKind::GenericParamListHint
            | InlayKind::ImplicitReborrowHint
            | InlayKind::TypeHint
            | InlayKind::ClosingBraceHint => false,
            InlayKind::BindingModeHint => inlay_hint.label.as_simple_str() != Some("&"),
            InlayKind::ParameterHint | InlayKind::LifetimeHint => true,
        }),
        kind: match inlay_hint.kind {
            InlayKind::ParameterHint => Some(lsp_types::InlayHintKind::PARAMETER),
            InlayKind::ClosureReturnTypeHint | InlayKind::TypeHint | InlayKind::ChainingHint => {
                Some(lsp_types::InlayHintKind::TYPE)
            }
            InlayKind::BindingModeHint
            | InlayKind::GenericParamListHint
            | InlayKind::LifetimeHint
            | InlayKind::ImplicitReborrowHint
            | InlayKind::ClosingBraceHint => None,
        },
        text_edits: None,
        data: (|| match inlay_hint.tooltip {
            Some(ide::InlayTooltip::HoverOffset(file_id, offset)) => {
                let uri = url(snap, file_id);
                let line_index = snap.file_line_index(file_id).ok()?;

                let text_document = lsp_types::TextDocumentIdentifier { uri };
                to_value(lsp_ext::InlayHintResolveData {
                    text_document,
                    position: lsp_ext::PositionOrRange::Position(position(&line_index, offset)),
                })
                .ok()
            }
            Some(ide::InlayTooltip::HoverRanged(file_id, text_range)) => {
                let uri = url(snap, file_id);
                let text_document = lsp_types::TextDocumentIdentifier { uri };
                let line_index = snap.file_line_index(file_id).ok()?;
                to_value(lsp_ext::InlayHintResolveData {
                    text_document,
                    position: lsp_ext::PositionOrRange::Range(range(&line_index, text_range)),
                })
                .ok()
            }
            _ => None,
        })(),
        tooltip: Some(match inlay_hint.tooltip {
            Some(ide::InlayTooltip::String(s)) => lsp_types::InlayHintTooltip::String(s),
            _ => lsp_types::InlayHintTooltip::String(inlay_hint.label.to_string()),
        }),
        label: inlay_hint_label(snap, inlay_hint.label)?,
    })
}

fn inlay_hint_label(
    snap: &GlobalStateSnapshot,
    label: InlayHintLabel,
) -> Result<lsp_types::InlayHintLabel> {
    Ok(match label.as_simple_str() {
        Some(s) => lsp_types::InlayHintLabel::String(s.into()),
        None => lsp_types::InlayHintLabel::LabelParts(
            label
                .parts
                .into_iter()
                .map(|part| {
                    Ok(lsp_types::InlayHintLabelPart {
                        value: part.text,
                        tooltip: None,
                        location: part
                            .linked_location
                            .map(|range| location(snap, range))
                            .transpose()?,
                        command: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ),
    })
}

static TOKEN_RESULT_COUNTER: AtomicU32 = AtomicU32::new(1);

pub(crate) fn semantic_tokens(
    text: &str,
    line_index: &LineIndex,
    highlights: Vec<HlRange>,
) -> lsp_types::SemanticTokens {
    let id = TOKEN_RESULT_COUNTER.fetch_add(1, Ordering::SeqCst).to_string();
    let mut builder = semantic_tokens::SemanticTokensBuilder::new(id);

    for highlight_range in highlights {
        if highlight_range.highlight.is_empty() {
            continue;
        }

        let (ty, mods) = semantic_token_type_and_modifiers(highlight_range.highlight);
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
            SymbolKind::Macro => semantic_tokens::MACRO,
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
        }
    } else {
        lsp_types::FoldingRange {
            start_line: range.start.line,
            start_character: Some(range.start.character),
            end_line: range.end.line,
            end_character: Some(range.end.character),
            kind,
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
) -> Result<lsp_types::Location> {
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
) -> Result<lsp_types::Location> {
    let url = url(snap, nav.file_id);
    let line_index = snap.file_line_index(nav.file_id)?;
    let range = range(&line_index, nav.full_range);
    let loc = lsp_types::Location::new(url, range);
    Ok(loc)
}

pub(crate) fn location_link(
    snap: &GlobalStateSnapshot,
    src: Option<FileRange>,
    target: NavigationTarget,
) -> Result<lsp_types::LocationLink> {
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
) -> Result<(lsp_types::Url, lsp_types::Range, lsp_types::Range)> {
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
) -> Result<lsp_types::GotoDefinitionResponse> {
    if snap.config.location_link() {
        let links = targets
            .into_iter()
            .map(|nav| location_link(snap, src, nav))
            .collect::<Result<Vec<_>>>()?;
        Ok(links.into())
    } else {
        let locations = targets
            .into_iter()
            .map(|nav| {
                location(snap, FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(locations.into())
    }
}

fn outside_workspace_annotation_id() -> String {
    String::from("OutsideWorkspace")
}

pub(crate) fn snippet_text_document_edit(
    snap: &GlobalStateSnapshot,
    is_snippet: bool,
    file_id: FileId,
    edit: TextEdit,
) -> Result<lsp_ext::SnippetTextDocumentEdit> {
    let text_document = optional_versioned_text_document_identifier(snap, file_id);
    let line_index = snap.file_line_index(file_id)?;
    let mut edits: Vec<_> =
        edit.into_iter().map(|it| snippet_text_edit(&line_index, is_snippet, it)).collect();

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
    source_change: SourceChange,
) -> Result<lsp_ext::SnippetWorkspaceEdit> {
    let mut document_changes: Vec<lsp_ext::SnippetDocumentChangeOperation> = Vec::new();

    for op in source_change.file_system_edits {
        let ops = snippet_text_document_ops(snap, op)?;
        document_changes.extend_from_slice(&ops);
    }
    for (file_id, edit) in source_change.source_file_edits {
        let edit = snippet_text_document_edit(snap, source_change.is_snippet, file_id, edit)?;
        document_changes.push(lsp_ext::SnippetDocumentChangeOperation::Edit(edit));
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
) -> Result<lsp_types::WorkspaceEdit> {
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
) -> Result<lsp_types::CallHierarchyItem> {
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
) -> Result<lsp_ext::CodeAction> {
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
                id: format!("{}:{}:{}", assist.id.0, assist.id.1.name(), index),
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
) -> Result<lsp_ext::Runnable> {
    let config = snap.config.runnables();
    let spec = CargoTargetSpec::for_file(snap, runnable.nav.file_id)?;
    let workspace_root = spec.as_ref().map(|it| it.workspace_root.clone());
    let target = spec.as_ref().map(|s| s.target.clone());
    let (cargo_args, executable_args) =
        CargoTargetSpec::runnable_args(snap, spec, &runnable.kind, &runnable.cfg)?;
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
) -> Result<()> {
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
            if lens_config.run && client_commands_config.run_single {
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
        }
        AnnotationKind::HasImpls { file_id, data } => {
            if !client_commands_config.show_reference {
                return Ok(());
            }
            let line_index = snap.file_line_index(file_id)?;
            let annotation_range = range(&line_index, annotation.range);
            let url = url(snap, file_id);

            let id = lsp_types::TextDocumentIdentifier { uri: url.clone() };

            let doc_pos = lsp_types::TextDocumentPositionParams::new(id, annotation_range.start);

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
                    annotation_range.start,
                    locations,
                )
            });

            acc.push(lsp_types::CodeLens {
                range: annotation_range,
                command,
                data: Some(to_value(lsp_ext::CodeLensResolveData::Impls(goto_params)).unwrap()),
            })
        }
        AnnotationKind::HasReferences { file_id, data } => {
            if !client_commands_config.show_reference {
                return Ok(());
            }
            let line_index = snap.file_line_index(file_id)?;
            let annotation_range = range(&line_index, annotation.range);
            let url = url(snap, file_id);

            let id = lsp_types::TextDocumentIdentifier { uri: url.clone() };

            let doc_pos = lsp_types::TextDocumentPositionParams::new(id, annotation_range.start);

            let command = data.map(|ranges| {
                let locations: Vec<lsp_types::Location> =
                    ranges.into_iter().filter_map(|range| location(snap, range).ok()).collect();

                command::show_references(
                    reference_title(locations.len()),
                    &url,
                    annotation_range.start,
                    locations,
                )
            });

            acc.push(lsp_types::CodeLens {
                range: annotation_range,
                command,
                data: Some(to_value(lsp_ext::CodeLensResolveData::References(doc_pos)).unwrap()),
            })
        }
    }
    Ok(())
}

pub(crate) mod command {
    use ide::{FileRange, NavigationTarget};
    use serde_json::to_value;

    use crate::{
        global_state::GlobalStateSnapshot,
        lsp_ext,
        to_proto::{location, location_link},
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
            title: title.to_string(),
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
            command: "editor.action.triggerParameterHints".into(),
            arguments: None,
        }
    }
}

pub(crate) fn implementation_title(count: usize) -> String {
    if count == 1 {
        "1 implementation".into()
    } else {
        format!("{} implementations", count)
    }
}

pub(crate) fn reference_title(count: usize) -> String {
    if count == 1 {
        "1 reference".into()
    } else {
        format!("{} references", count)
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
    let value = crate::markdown::format_docs(markup.as_str());
    lsp_types::MarkupContent { kind, value }
}

pub(crate) fn rename_error(err: RenameError) -> crate::LspError {
    // This is wrong, but we don't have a better alternative I suppose?
    // https://github.com/microsoft/language-server-protocol/issues/1341
    invalid_params_error(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ide::Analysis;

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

        let (analysis, file_id) = Analysis::from_single_file(text.to_string());
        let folds = analysis.folding_ranges(file_id).unwrap();
        assert_eq!(folds.len(), 4);

        let line_index = LineIndex {
            index: Arc::new(ide::LineIndex::new(text)),
            endings: LineEndings::Unix,
            encoding: OffsetEncoding::Utf16,
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
