//! Advertizes the capabilities of the LSP Server.
use std::env;

use lsp_types::{
    CallHierarchyServerCapability, ClientCapabilities, CodeActionKind, CodeActionOptions,
    CodeActionProviderCapability, CodeLensOptions, CompletionOptions,
    DocumentOnTypeFormattingOptions, FoldingRangeProviderCapability, HoverProviderCapability,
    ImplementationProviderCapability, RenameOptions, RenameProviderCapability, SaveOptions,
    SelectionRangeProviderCapability, SemanticTokensDocumentProvider, SemanticTokensLegend,
    SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities,
    SignatureHelpOptions, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, TypeDefinitionProviderCapability, WorkDoneProgressOptions,
};
use serde_json::{json, Value};

use crate::semantic_tokens;

pub fn server_capabilities(client_caps: &ClientCapabilities) -> ServerCapabilities {
    let code_action_provider = code_action_capabilities(client_caps);
    let semantic_tokens_provider = semantic_tokens_capabilities(client_caps);
    let experimental = experimental_capabilities(client_caps);

    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(if env::var("RA_NO_INCREMENTAL_SYNC").is_ok() {
                TextDocumentSyncKind::Full
            } else {
                TextDocumentSyncKind::Incremental
            }),
            will_save: None,
            will_save_wait_until: None,
            save: Some(SaveOptions::default().into()),
        })),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        completion_provider: Some(CompletionOptions {
            resolve_provider: None,
            trigger_characters: Some(vec![":".to_string(), ".".to_string()]),
            work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        }),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
            retrigger_characters: None,
            work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        }),
        declaration_provider: None,
        definition_provider: Some(true),
        type_definition_provider: Some(TypeDefinitionProviderCapability::Simple(true)),
        implementation_provider: Some(ImplementationProviderCapability::Simple(true)),
        references_provider: Some(true),
        document_highlight_provider: Some(true),
        document_symbol_provider: Some(true),
        workspace_symbol_provider: Some(true),
        code_action_provider: Some(code_action_provider),
        code_lens_provider: Some(CodeLensOptions { resolve_provider: Some(true) }),
        document_formatting_provider: Some(true),
        document_range_formatting_provider: None,
        document_on_type_formatting_provider: Some(DocumentOnTypeFormattingOptions {
            first_trigger_character: "=".to_string(),
            more_trigger_character: Some(vec![".".to_string(), ">".to_string()]),
        }),
        selection_range_provider: Some(SelectionRangeProviderCapability::Simple(true)),
        semantic_highlighting: None,
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        rename_provider: Some(RenameProviderCapability::Options(RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        })),
        document_link_provider: None,
        color_provider: None,
        execute_command_provider: None,
        workspace: None,
        call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
        semantic_tokens_provider,
        experimental,
    }
}

fn experimental_capabilities(client_caps: &ClientCapabilities) -> Option<Value> {
    client_caps.experimental.as_ref().and_then(|it| {
        it.as_object().map(|map| {
            let mut obj = json!({});
            let result = obj.as_object_mut().unwrap();

            if map.contains_key("joinLines") {
                result.insert("joinLines".into(), true.into());
            }

            if map.contains_key("ssr") {
                result.insert("ssr".into(), true.into());
            }

            if map.contains_key("onEnter") {
                result.insert("onEnter".into(), true.into());
            }

            if map.contains_key("parentModule") {
                result.insert("parentModule".into(), true.into());
            }

            if map.contains_key("runnables") {
                result.insert("runnables".into(), json!({ "kinds": [ "cargo" ] }));
            }

            obj
        })
    })
}

fn semantic_tokens_capabilities(
    client_caps: &ClientCapabilities,
) -> Option<SemanticTokensServerCapabilities> {
    client_caps.text_document.as_ref().and_then(|it| it.semantic_tokens.as_ref()).map(|_|
            // client supports semanticTokens
            SemanticTokensOptions {
            legend: SemanticTokensLegend {
                token_types: semantic_tokens::SUPPORTED_TYPES.to_vec(),
                token_modifiers: semantic_tokens::SUPPORTED_MODIFIERS.to_vec(),
            },

            document_provider: Some(SemanticTokensDocumentProvider::Bool(true)),
            range_provider: Some(true),
            work_done_progress_options: Default::default(),
        }
        .into())
}

fn code_action_capabilities(client_caps: &ClientCapabilities) -> CodeActionProviderCapability {
    client_caps
        .text_document
        .as_ref()
        .and_then(|it| it.code_action.as_ref())
        .and_then(|it| it.code_action_literal_support.as_ref())
        .map_or(CodeActionProviderCapability::Simple(true), |caps| {
            let mut action_kinds = vec![
                CodeActionKind::EMPTY,
                CodeActionKind::QUICKFIX,
                CodeActionKind::REFACTOR,
                CodeActionKind::REFACTOR_EXTRACT,
                CodeActionKind::REFACTOR_INLINE,
                CodeActionKind::REFACTOR_REWRITE,
            ];

            // Not all clients can fall back gracefully for unknown values.
            // Microsoft.VisualStudio.LanguageServer.Protocol.CodeActionKind does not support CodeActionKind::EMPTY
            // So have to filter out.
            action_kinds
                .retain(|it| caps.code_action_kind.value_set.contains(&it.as_str().to_owned()));

            CodeActionProviderCapability::Options(CodeActionOptions {
                code_action_kinds: Some(action_kinds),
                work_done_progress_options: Default::default(),
            })
        })
}
