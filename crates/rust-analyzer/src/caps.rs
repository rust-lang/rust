//! Advertizes the capabilities of the LSP Server.

use lsp_types::{
    CallHierarchyServerCapability, CodeActionProviderCapability, CodeLensOptions,
    CompletionOptions, DocumentOnTypeFormattingOptions, FoldingRangeProviderCapability,
    ImplementationProviderCapability, RenameOptions, RenameProviderCapability, SaveOptions,
    SelectionRangeProviderCapability, ServerCapabilities, SignatureHelpOptions,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
    TypeDefinitionProviderCapability, WorkDoneProgressOptions,
};

pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::Full),
            will_save: None,
            will_save_wait_until: None,
            save: Some(SaveOptions::default()),
        })),
        hover_provider: Some(true),
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
        code_action_provider: Some(CodeActionProviderCapability::Simple(true)),
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
        experimental: Default::default(),
    }
}
