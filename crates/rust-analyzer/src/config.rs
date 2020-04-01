//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use lsp_types::TextDocumentClientCapabilities;
use ra_flycheck::FlycheckConfig;
use ra_ide::{CompletionConfig, InlayHintsConfig};
use ra_project_model::CargoConfig;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Config {
    pub client_caps: ClientCapsConfig,
    pub publish_decorations: bool,
    pub publish_diagnostics: bool,
    pub notifications: NotificationsConfig,
    pub inlay_hints: InlayHintsConfig,
    pub completion: CompletionConfig,
    pub call_info_full: bool,
    pub rustfmt: RustfmtConfig,
    pub check: Option<FlycheckConfig>,
    pub vscode_lldb: bool,
    pub proc_macro_srv: Option<String>,
    pub lru_capacity: Option<usize>,
    pub use_client_watching: bool,
    pub exclude_globs: Vec<String>,
    pub cargo: CargoConfig,
    pub with_sysroot: bool,
}

#[derive(Debug, Clone)]
pub struct NotificationsConfig {
    pub workspace_loaded: bool,
    pub cargo_toml_not_found: bool,
}

#[derive(Debug, Clone)]
pub enum RustfmtConfig {
    Rustfmt {
        extra_args: Vec<String>,
    },
    #[allow(unused)]
    CustomCommand {
        command: String,
        args: Vec<String>,
    },
}

impl Default for RustfmtConfig {
    fn default() -> Self {
        RustfmtConfig::Rustfmt { extra_args: Vec::new() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ClientCapsConfig {
    pub location_link: bool,
    pub line_folding_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            publish_decorations: false,
            publish_diagnostics: true,
            notifications: NotificationsConfig {
                workspace_loaded: true,
                cargo_toml_not_found: true,
            },
            client_caps: ClientCapsConfig::default(),
            inlay_hints: InlayHintsConfig {
                type_hints: true,
                parameter_hints: true,
                chaining_hints: true,
                max_length: None,
            },
            completion: CompletionConfig {
                enable_postfix_completions: true,
                add_call_parenthesis: true,
                add_call_argument_snippets: true,
            },
            call_info_full: true,
            rustfmt: RustfmtConfig::default(),
            check: Some(FlycheckConfig::default()),
            vscode_lldb: false,
            proc_macro_srv: None,
            lru_capacity: None,
            use_client_watching: false,
            exclude_globs: Vec::new(),
            cargo: CargoConfig::default(),
            with_sysroot: true,
        }
    }
}

impl Config {
    #[rustfmt::skip]
    pub fn update(&mut self, value: &serde_json::Value) {
        let client_caps = self.client_caps.clone();
        *self = Default::default();
        self.client_caps = client_caps;

        set(value, "publishDecorations", &mut self.publish_decorations);
        set(value, "excludeGlobs", &mut self.exclude_globs);
        set(value, "useClientWatching", &mut self.use_client_watching);
        set(value, "lruCapacity", &mut self.lru_capacity);

        set(value, "inlayHintsType", &mut self.inlay_hints.type_hints);
        set(value, "inlayHintsParameter", &mut self.inlay_hints.parameter_hints);
        set(value, "inlayHintsChaining", &mut self.inlay_hints.chaining_hints);
        set(value, "inlayHintsMaxLength", &mut self.inlay_hints.max_length);

        if let Some(false) = get(value, "cargo_watch_enable") {
            self.check = None
        } else {
            if let Some(FlycheckConfig::CargoCommand { command, extra_args, all_targets }) = &mut self.check
            {
                set(value, "cargoWatchArgs", extra_args);
                set(value, "cargoWatchCommand", command);
                set(value, "cargoWatchAllTargets", all_targets);
            }
        };

        set(value, "withSysroot", &mut self.with_sysroot);
        if let RustfmtConfig::Rustfmt { extra_args } = &mut self.rustfmt {
            set(value, "rustfmtArgs", extra_args);
        }

        set(value, "cargoFeatures/noDefaultFeatures", &mut self.cargo.no_default_features);
        set(value, "cargoFeatures/allFeatures", &mut self.cargo.all_features);
        set(value, "cargoFeatures/features", &mut self.cargo.features);
        set(value, "cargoFeatures/loadOutDirsFromCheck", &mut self.cargo.load_out_dirs_from_check);

        set(value, "vscodeLldb", &mut self.vscode_lldb);

        set(value, "featureFlags/lsp.diagnostics", &mut self.publish_diagnostics);
        set(value, "featureFlags/notifications.workspace-loaded", &mut self.notifications.workspace_loaded);
        set(value, "featureFlags/notifications.cargo-toml-not-found", &mut self.notifications.cargo_toml_not_found);
        set(value, "featureFlags/completion.enable-postfix", &mut self.completion.enable_postfix_completions);
        set(value, "featureFlags/completion.insertion.add-call-parenthesis", &mut self.completion.add_call_parenthesis);
        set(value, "featureFlags/completion.insertion.add-argument-snippets", &mut self.completion.add_call_argument_snippets);
        set(value, "featureFlags/call-info.full", &mut self.call_info_full);

        fn get<'a, T: Deserialize<'a>>(value: &'a serde_json::Value, pointer: &str) -> Option<T> {
            value.pointer(pointer).and_then(|it| T::deserialize(it).ok())
        }

        fn set<'a, T: Deserialize<'a>>(value: &'a serde_json::Value, pointer: &str, slot: &mut T) {
            if let Some(new_value) = get(value, pointer) {
                *slot = new_value
            }
        }
    }

    pub fn update_caps(&mut self, caps: &TextDocumentClientCapabilities) {
        if let Some(value) = caps.definition.as_ref().and_then(|it| it.link_support) {
            self.client_caps.location_link = value;
        }
        if let Some(value) = caps.folding_range.as_ref().and_then(|it| it.line_folding_only) {
            self.client_caps.line_folding_only = value
        }
    }
}
