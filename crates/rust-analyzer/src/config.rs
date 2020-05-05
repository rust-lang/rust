//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use std::{ffi::OsString, path::PathBuf};

use lsp_types::TextDocumentClientCapabilities;
use ra_flycheck::FlycheckConfig;
use ra_ide::{CompletionConfig, InlayHintsConfig};
use ra_project_model::CargoConfig;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Config {
    pub client_caps: ClientCapsConfig,

    pub with_sysroot: bool,
    pub publish_diagnostics: bool,
    pub lru_capacity: Option<usize>,
    pub proc_macro_srv: Option<(PathBuf, Vec<OsString>)>,
    pub files: FilesConfig,
    pub notifications: NotificationsConfig,

    pub cargo: CargoConfig,
    pub rustfmt: RustfmtConfig,
    pub check: Option<FlycheckConfig>,

    pub inlay_hints: InlayHintsConfig,
    pub completion: CompletionConfig,
    pub call_info_full: bool,
}

#[derive(Debug, Clone)]
pub struct FilesConfig {
    pub watcher: FilesWatcher,
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum FilesWatcher {
    Client,
    Notify,
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

#[derive(Debug, Clone, Default)]
pub struct ClientCapsConfig {
    pub location_link: bool,
    pub line_folding_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            client_caps: ClientCapsConfig::default(),

            with_sysroot: true,
            publish_diagnostics: true,
            lru_capacity: None,
            proc_macro_srv: None,
            files: FilesConfig { watcher: FilesWatcher::Notify, exclude: Vec::new() },
            notifications: NotificationsConfig {
                workspace_loaded: true,
                cargo_toml_not_found: true,
            },

            cargo: CargoConfig::default(),
            rustfmt: RustfmtConfig::Rustfmt { extra_args: Vec::new() },
            check: Some(FlycheckConfig::CargoCommand {
                command: "check".to_string(),
                all_targets: true,
                extra_args: Vec::new(),
            }),

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
                ..CompletionConfig::default()
            },
            call_info_full: true,
        }
    }
}

impl Config {
    #[rustfmt::skip]
    pub fn update(&mut self, value: &serde_json::Value) {
        log::info!("Config::update({:#})", value);

        let client_caps = self.client_caps.clone();
        *self = Default::default();
        self.client_caps = client_caps;

        set(value, "/withSysroot", &mut self.with_sysroot);
        set(value, "/diagnostics/enable", &mut self.publish_diagnostics);
        set(value, "/lruCapacity", &mut self.lru_capacity);
        self.files.watcher = match get(value, "/files/watcher") {
            Some("client") => FilesWatcher::Client,
            Some("notify") | _ => FilesWatcher::Notify
        };
        set(value, "/notifications/workspaceLoaded", &mut self.notifications.workspace_loaded);
        set(value, "/notifications/cargoTomlNotFound", &mut self.notifications.cargo_toml_not_found);

        set(value, "/cargo/noDefaultFeatures", &mut self.cargo.no_default_features);
        set(value, "/cargo/allFeatures", &mut self.cargo.all_features);
        set(value, "/cargo/features", &mut self.cargo.features);
        set(value, "/cargo/loadOutDirsFromCheck", &mut self.cargo.load_out_dirs_from_check);
        set(value, "/cargo/target", &mut self.cargo.target);

        match get(value, "/procMacro/enable") {
            Some(true) => {
                if let Ok(path) = std::env::current_exe() {
                    self.proc_macro_srv = Some((path, vec!["proc-macro".into()]));
                }
            }
            _ => self.proc_macro_srv = None,
        }

        match get::<Vec<String>>(value, "/rustfmt/overrideCommand") {
            Some(mut args) if !args.is_empty() => {
                let command = args.remove(0);
                self.rustfmt = RustfmtConfig::CustomCommand {
                    command,
                    args,
                }
            }
            _ => {
                if let RustfmtConfig::Rustfmt { extra_args } = &mut self.rustfmt {
                    set(value, "/rustfmt/extraArgs", extra_args);
                }
            }
        };

        if let Some(false) = get(value, "/checkOnSave/enable") {
            // check is disabled
            self.check = None;
        } else {
            // check is enabled
            match get::<Vec<String>>(value, "/checkOnSave/overrideCommand") {
                // first see if the user has completely overridden the command
                Some(mut args) if !args.is_empty() => {
                    let command = args.remove(0);
                    self.check = Some(FlycheckConfig::CustomCommand {
                        command,
                        args,
                    });
                }
                // otherwise configure command customizations
                _ => {
                    if let Some(FlycheckConfig::CargoCommand { command, extra_args, all_targets })
                        = &mut self.check
                    {
                        set(value, "/checkOnSave/extraArgs", extra_args);
                        set(value, "/checkOnSave/command", command);
                        set(value, "/checkOnSave/allTargets", all_targets);
                    }
                }
            };
        }

        set(value, "/inlayHints/typeHints", &mut self.inlay_hints.type_hints);
        set(value, "/inlayHints/parameterHints", &mut self.inlay_hints.parameter_hints);
        set(value, "/inlayHints/chainingHints", &mut self.inlay_hints.chaining_hints);
        set(value, "/inlayHints/maxLength", &mut self.inlay_hints.max_length);
        set(value, "/completion/postfix/enable", &mut self.completion.enable_postfix_completions);
        set(value, "/completion/addCallParenthesis", &mut self.completion.add_call_parenthesis);
        set(value, "/completion/addCallArgumentSnippets", &mut self.completion.add_call_argument_snippets);
        set(value, "/callInfo/full", &mut self.call_info_full);

        log::info!("Config::update() = {:#?}", self);

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
        self.completion.allow_snippets(false);
        if let Some(completion) = &caps.completion {
            if let Some(completion_item) = &completion.completion_item {
                if let Some(value) = completion_item.snippet_support {
                    self.completion.allow_snippets(value);
                }
            }
        }
    }
}
