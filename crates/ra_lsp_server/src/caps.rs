//! FIXME: write short doc here

use lsp_types::{
    CodeActionProviderCapability, CodeLensOptions, CompletionOptions,
    DocumentOnTypeFormattingOptions, FoldingRangeProviderCapability, GenericCapability,
    ImplementationProviderCapability, RenameOptions, RenameProviderCapability, ServerCapabilities,
    SignatureHelpOptions, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, TypeDefinitionProviderCapability,
};

pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::Full),
            will_save: None,
            will_save_wait_until: None,
            save: None,
        })),
        hover_provider: Some(true),
        completion_provider: Some(CompletionOptions {
            resolve_provider: None,
            trigger_characters: Some(vec![":".to_string(), ".".to_string()]),
        }),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string(), ")".to_string()]),
        }),
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
        selection_range_provider: Some(GenericCapability::default()),
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        rename_provider: Some(RenameProviderCapability::Options(RenameOptions {
            prepare_provider: Some(true),
        })),
        document_link_provider: None,
        color_provider: None,
        execute_command_provider: None,
        workspace: None,
    }
}
