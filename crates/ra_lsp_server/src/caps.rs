use languageserver_types::{
    ServerCapabilities,
    CodeActionProviderCapability,
    FoldingRangeProviderCapability,
    TextDocumentSyncCapability,
    TextDocumentSyncOptions,
    TextDocumentSyncKind,
    ExecuteCommandOptions,
    CompletionOptions,
    DocumentOnTypeFormattingOptions,
};

pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Options(
            TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(TextDocumentSyncKind::Full),
                will_save: None,
                will_save_wait_until: None,
                save: None,
            }
        )),
        hover_provider: None,
        completion_provider: Some(CompletionOptions {
            resolve_provider: None,
            trigger_characters: None,
        }),
        signature_help_provider: None,
        definition_provider: Some(true),
        type_definition_provider: None,
        implementation_provider: None,
        references_provider: None,
        document_highlight_provider: None,
        document_symbol_provider: Some(true),
        workspace_symbol_provider: Some(true),
        code_action_provider: Some(CodeActionProviderCapability::Simple(true)),
        code_lens_provider: None,
        document_formatting_provider: None,
        document_range_formatting_provider: None,
        document_on_type_formatting_provider: Some(DocumentOnTypeFormattingOptions {
            first_trigger_character: "=".to_string(),
            more_trigger_character: None,
        }),
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        rename_provider: None,
        color_provider: None,
        execute_command_provider: Some(ExecuteCommandOptions {
            commands: vec!["apply_code_action".to_string()],
        }),
        workspace: None,
    }
}
