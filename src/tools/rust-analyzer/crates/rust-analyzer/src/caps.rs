//! Advertises the capabilities of the LSP Server.
use lsp_types::{
    CallHierarchyServerCapability, ClientCapabilities, CodeActionKind, CodeActionOptions,
    CodeActionProviderCapability, CodeLensOptions, CompletionOptions,
    CompletionOptionsCompletionItem, DeclarationCapability, DocumentOnTypeFormattingOptions,
    FileOperationFilter, FileOperationPattern, FileOperationPatternKind,
    FileOperationRegistrationOptions, FoldingRangeProviderCapability, HoverProviderCapability,
    ImplementationProviderCapability, InlayHintOptions, InlayHintServerCapabilities, OneOf,
    PositionEncodingKind, RenameOptions, SaveOptions, SelectionRangeProviderCapability,
    SemanticTokensFullOptions, SemanticTokensLegend, SemanticTokensOptions, ServerCapabilities,
    SignatureHelpOptions, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, TypeDefinitionProviderCapability, WorkDoneProgressOptions,
    WorkspaceFileOperationsServerCapabilities, WorkspaceFoldersServerCapabilities,
    WorkspaceServerCapabilities,
};
use serde_json::json;

use crate::config::{Config, RustfmtConfig};
use crate::lsp_ext::supports_utf8;
use crate::semantic_tokens;

pub fn server_capabilities(config: &Config) -> ServerCapabilities {
    ServerCapabilities {
        position_encoding: if supports_utf8(config.caps()) {
            Some(PositionEncodingKind::UTF8)
        } else {
            None
        },
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::INCREMENTAL),
            will_save: None,
            will_save_wait_until: None,
            save: Some(SaveOptions::default().into()),
        })),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        completion_provider: Some(CompletionOptions {
            resolve_provider: completions_resolve_provider(config.caps()),
            trigger_characters: Some(vec![
                ":".to_string(),
                ".".to_string(),
                "'".to_string(),
                "(".to_string(),
            ]),
            all_commit_characters: None,
            completion_item: completion_item(config),
            work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        }),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string(), "<".to_string()]),
            retrigger_characters: None,
            work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        }),
        declaration_provider: Some(DeclarationCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        type_definition_provider: Some(TypeDefinitionProviderCapability::Simple(true)),
        implementation_provider: Some(ImplementationProviderCapability::Simple(true)),
        references_provider: Some(OneOf::Left(true)),
        document_highlight_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
        code_action_provider: Some(code_action_capabilities(config.caps())),
        code_lens_provider: Some(CodeLensOptions { resolve_provider: Some(true) }),
        document_formatting_provider: Some(OneOf::Left(true)),
        document_range_formatting_provider: match config.rustfmt() {
            RustfmtConfig::Rustfmt { enable_range_formatting: true, .. } => Some(OneOf::Left(true)),
            _ => Some(OneOf::Left(false)),
        },
        document_on_type_formatting_provider: Some(DocumentOnTypeFormattingOptions {
            first_trigger_character: "=".to_string(),
            more_trigger_character: Some(more_trigger_character(config)),
        }),
        selection_range_provider: Some(SelectionRangeProviderCapability::Simple(true)),
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        rename_provider: Some(OneOf::Right(RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        })),
        linked_editing_range_provider: None,
        document_link_provider: None,
        color_provider: None,
        execute_command_provider: None,
        workspace: Some(WorkspaceServerCapabilities {
            workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(OneOf::Left(true)),
            }),
            file_operations: Some(WorkspaceFileOperationsServerCapabilities {
                did_create: None,
                will_create: None,
                did_rename: None,
                will_rename: Some(FileOperationRegistrationOptions {
                    filters: vec![
                        FileOperationFilter {
                            scheme: Some(String::from("file")),
                            pattern: FileOperationPattern {
                                glob: String::from("**/*.rs"),
                                matches: Some(FileOperationPatternKind::File),
                                options: None,
                            },
                        },
                        FileOperationFilter {
                            scheme: Some(String::from("file")),
                            pattern: FileOperationPattern {
                                glob: String::from("**"),
                                matches: Some(FileOperationPatternKind::Folder),
                                options: None,
                            },
                        },
                    ],
                }),
                did_delete: None,
                will_delete: None,
            }),
        }),
        call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
        semantic_tokens_provider: Some(
            SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types: semantic_tokens::SUPPORTED_TYPES.to_vec(),
                    token_modifiers: semantic_tokens::SUPPORTED_MODIFIERS.to_vec(),
                },

                full: Some(SemanticTokensFullOptions::Delta { delta: Some(true) }),
                range: Some(true),
                work_done_progress_options: Default::default(),
            }
            .into(),
        ),
        moniker_provider: None,
        inlay_hint_provider: Some(OneOf::Right(InlayHintServerCapabilities::Options(
            InlayHintOptions {
                work_done_progress_options: Default::default(),
                resolve_provider: Some(true),
            },
        ))),
        experimental: Some(json!({
            "externalDocs": true,
            "hoverRange": true,
            "joinLines": true,
            "matchingBrace": true,
            "moveItem": true,
            "onEnter": true,
            "openCargoToml": true,
            "parentModule": true,
            "runnables": {
                "kinds": [ "cargo" ],
            },
            "ssr": true,
            "workspaceSymbolScopeKindFiltering": true,
        })),
    }
}

fn completions_resolve_provider(client_caps: &ClientCapabilities) -> Option<bool> {
    if completion_item_edit_resolve(client_caps) {
        Some(true)
    } else {
        tracing::info!("No `additionalTextEdits` completion resolve capability was found in the client capabilities, autoimport completion is disabled");
        None
    }
}

/// Parses client capabilities and returns all completion resolve capabilities rust-analyzer supports.
pub(crate) fn completion_item_edit_resolve(caps: &ClientCapabilities) -> bool {
    (|| {
        Some(
            caps.text_document
                .as_ref()?
                .completion
                .as_ref()?
                .completion_item
                .as_ref()?
                .resolve_support
                .as_ref()?
                .properties
                .iter()
                .any(|cap_string| cap_string.as_str() == "additionalTextEdits"),
        )
    })() == Some(true)
}

fn completion_item(config: &Config) -> Option<CompletionOptionsCompletionItem> {
    Some(CompletionOptionsCompletionItem {
        label_details_support: Some(config.completion_label_details_support()),
    })
}

fn code_action_capabilities(client_caps: &ClientCapabilities) -> CodeActionProviderCapability {
    client_caps
        .text_document
        .as_ref()
        .and_then(|it| it.code_action.as_ref())
        .and_then(|it| it.code_action_literal_support.as_ref())
        .map_or(CodeActionProviderCapability::Simple(true), |_| {
            CodeActionProviderCapability::Options(CodeActionOptions {
                // Advertise support for all built-in CodeActionKinds.
                // Ideally we would base this off of the client capabilities
                // but the client is supposed to fall back gracefully for unknown values.
                code_action_kinds: Some(vec![
                    CodeActionKind::EMPTY,
                    CodeActionKind::QUICKFIX,
                    CodeActionKind::REFACTOR,
                    CodeActionKind::REFACTOR_EXTRACT,
                    CodeActionKind::REFACTOR_INLINE,
                    CodeActionKind::REFACTOR_REWRITE,
                ]),
                resolve_provider: Some(true),
                work_done_progress_options: Default::default(),
            })
        })
}

fn more_trigger_character(config: &Config) -> Vec<String> {
    let mut res = vec![".".to_string(), ">".to_string(), "{".to_string()];
    if config.snippet_cap() {
        res.push("<".to_string());
    }
    res
}
