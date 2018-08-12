use languageserver_types::{
    ServerCapabilities,
    TextDocumentSyncCapability,
    TextDocumentSyncOptions,
    TextDocumentSyncKind,
};

pub const SERVER_CAPABILITIES: ServerCapabilities = ServerCapabilities {
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
    completion_provider: None,
    signature_help_provider: None,
    definition_provider: None,
    type_definition_provider: None,
    implementation_provider: None,
    references_provider: None,
    document_highlight_provider: None,
    document_symbol_provider: Some(true),
    workspace_symbol_provider: None,
    code_action_provider: Some(true),
    code_lens_provider: None,
    document_formatting_provider: None,
    document_range_formatting_provider: None,
    document_on_type_formatting_provider: None,
    rename_provider: None,
    color_provider: None,
    execute_command_provider: None,
};
