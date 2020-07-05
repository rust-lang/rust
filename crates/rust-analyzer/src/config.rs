//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use std::{ffi::OsString, path::PathBuf};

use flycheck::FlycheckConfig;
use lsp_types::ClientCapabilities;
use ra_db::AbsPathBuf;
use ra_ide::{AssistConfig, CompletionConfig, HoverConfig, InlayHintsConfig};
use ra_project_model::{CargoConfig, ProjectJson, ProjectJsonData, ProjectManifest};
use serde::Deserialize;

use crate::diagnostics::DiagnosticsConfig;

#[derive(Debug, Clone)]
pub struct Config {
    pub client_caps: ClientCapsConfig,

    pub publish_diagnostics: bool,
    pub diagnostics: DiagnosticsConfig,
    pub lru_capacity: Option<usize>,
    pub proc_macro_srv: Option<(PathBuf, Vec<OsString>)>,
    pub files: FilesConfig,
    pub notifications: NotificationsConfig,

    pub cargo: CargoConfig,
    pub rustfmt: RustfmtConfig,
    pub flycheck: Option<FlycheckConfig>,

    pub inlay_hints: InlayHintsConfig,
    pub completion: CompletionConfig,
    pub assist: AssistConfig,
    pub call_info_full: bool,
    pub lens: LensConfig,
    pub hover: HoverConfig,

    pub with_sysroot: bool,
    pub linked_projects: Vec<LinkedProject>,
    pub root_path: AbsPathBuf,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LinkedProject {
    ProjectManifest(ProjectManifest),
    InlineJsonProject(ProjectJson),
}

impl From<ProjectManifest> for LinkedProject {
    fn from(v: ProjectManifest) -> Self {
        LinkedProject::ProjectManifest(v)
    }
}

impl From<ProjectJson> for LinkedProject {
    fn from(v: ProjectJson) -> Self {
        LinkedProject::InlineJsonProject(v)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LensConfig {
    pub run: bool,
    pub debug: bool,
    pub implementations: bool,
}

impl Default for LensConfig {
    fn default() -> Self {
        Self { run: true, debug: true, implementations: true }
    }
}

impl LensConfig {
    pub const NO_LENS: LensConfig = Self { run: false, debug: false, implementations: false };

    pub fn any(&self) -> bool {
        self.implementations || self.runnable()
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn runnable(&self) -> bool {
        self.run || self.debug
    }
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
    pub hierarchical_symbols: bool,
    pub code_action_literals: bool,
    pub work_done_progress: bool,
    pub code_action_group: bool,
    pub resolve_code_action: bool,
    pub hover_actions: bool,
    pub status_notification: bool,
}

impl Config {
    pub fn new(root_path: AbsPathBuf) -> Self {
        Config {
            client_caps: ClientCapsConfig::default(),

            with_sysroot: true,
            publish_diagnostics: true,
            diagnostics: DiagnosticsConfig::default(),
            lru_capacity: None,
            proc_macro_srv: None,
            files: FilesConfig { watcher: FilesWatcher::Notify, exclude: Vec::new() },
            notifications: NotificationsConfig { cargo_toml_not_found: true },

            cargo: CargoConfig::default(),
            rustfmt: RustfmtConfig::Rustfmt { extra_args: Vec::new() },
            flycheck: Some(FlycheckConfig::CargoCommand {
                command: "check".to_string(),
                all_targets: true,
                all_features: false,
                extra_args: Vec::new(),
                features: Vec::new(),
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
            assist: AssistConfig::default(),
            call_info_full: true,
            lens: LensConfig::default(),
            hover: HoverConfig::default(),
            linked_projects: Vec::new(),
            root_path,
        }
    }

    #[rustfmt::skip]
    pub fn update(&mut self, value: &serde_json::Value) {
        log::info!("Config::update({:#})", value);

        let client_caps = self.client_caps.clone();
        let linked_projects = self.linked_projects.clone();
        *self = Config::new(self.root_path.clone());
        self.client_caps = client_caps;
        self.linked_projects = linked_projects;

        set(value, "/withSysroot", &mut self.with_sysroot);
        set(value, "/diagnostics/enable", &mut self.publish_diagnostics);
        set(value, "/diagnostics/warningsAsInfo", &mut self.diagnostics.warnings_as_info);
        set(value, "/diagnostics/warningsAsHint", &mut self.diagnostics.warnings_as_hint);
        set(value, "/lruCapacity", &mut self.lru_capacity);
        self.files.watcher = match get(value, "/files/watcher") {
            Some("client") => FilesWatcher::Client,
            Some("notify") | _ => FilesWatcher::Notify
        };
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
            self.flycheck = None;
        } else {
            // check is enabled
            match get::<Vec<String>>(value, "/checkOnSave/overrideCommand") {
                // first see if the user has completely overridden the command
                Some(mut args) if !args.is_empty() => {
                    let command = args.remove(0);
                    self.flycheck = Some(FlycheckConfig::CustomCommand {
                        command,
                        args,
                    });
                }
                // otherwise configure command customizations
                _ => {
                    if let Some(FlycheckConfig::CargoCommand { command, extra_args, all_targets, all_features, features })
                        = &mut self.flycheck
                    {
                        set(value, "/checkOnSave/extraArgs", extra_args);
                        set(value, "/checkOnSave/command", command);
                        set(value, "/checkOnSave/allTargets", all_targets);
                        *all_features = get(value, "/checkOnSave/allFeatures").unwrap_or(self.cargo.all_features);
                        *features = get(value, "/checkOnSave/features").unwrap_or(self.cargo.features.clone());
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

        let mut lens_enabled = true;
        set(value, "/lens/enable", &mut lens_enabled);
        if lens_enabled {
            set(value, "/lens/run", &mut self.lens.run);
            set(value, "/lens/debug", &mut self.lens.debug);
            set(value, "/lens/implementations", &mut self.lens.implementations);
        } else {
            self.lens = LensConfig::NO_LENS;
        }

        if let Some(linked_projects) = get::<Vec<ManifestOrProjectJson>>(value, "/linkedProjects") {
            if !linked_projects.is_empty() {
                self.linked_projects.clear();
                for linked_project in linked_projects {
                    let linked_project = match linked_project {
                        ManifestOrProjectJson::Manifest(it) => {
                            let path = self.root_path.join(it);
                            match ProjectManifest::from_manifest_file(path) {
                                Ok(it) => it.into(),
                                Err(_) => continue,
                            }
                        }
                        ManifestOrProjectJson::ProjectJson(it) => ProjectJson::new(&self.root_path, it).into(),
                    };
                    self.linked_projects.push(linked_project);
                }
            }
        }

        let mut use_hover_actions = false;
        set(value, "/hoverActions/enable", &mut use_hover_actions);
        if use_hover_actions {
            set(value, "/hoverActions/implementations", &mut self.hover.implementations);
            set(value, "/hoverActions/run", &mut self.hover.run);
            set(value, "/hoverActions/debug", &mut self.hover.debug);
            set(value, "/hoverActions/gotoTypeDef", &mut self.hover.goto_type_def);
        } else {
            self.hover = HoverConfig::NO_ACTIONS;
        }

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

    pub fn update_caps(&mut self, caps: &ClientCapabilities) {
        if let Some(doc_caps) = caps.text_document.as_ref() {
            if let Some(value) = doc_caps.definition.as_ref().and_then(|it| it.link_support) {
                self.client_caps.location_link = value;
            }
            if let Some(value) = doc_caps.folding_range.as_ref().and_then(|it| it.line_folding_only)
            {
                self.client_caps.line_folding_only = value
            }
            if let Some(value) = doc_caps
                .document_symbol
                .as_ref()
                .and_then(|it| it.hierarchical_document_symbol_support)
            {
                self.client_caps.hierarchical_symbols = value
            }
            if let Some(value) =
                doc_caps.code_action.as_ref().map(|it| it.code_action_literal_support.is_some())
            {
                self.client_caps.code_action_literals = value;
            }

            self.completion.allow_snippets(false);
            if let Some(completion) = &doc_caps.completion {
                if let Some(completion_item) = &completion.completion_item {
                    if let Some(value) = completion_item.snippet_support {
                        self.completion.allow_snippets(value);
                    }
                }
            }
        }

        if let Some(window_caps) = caps.window.as_ref() {
            if let Some(value) = window_caps.work_done_progress {
                self.client_caps.work_done_progress = value;
            }
        }

        self.assist.allow_snippets(false);
        if let Some(experimental) = &caps.experimental {
            let get_bool =
                |index: &str| experimental.get(index).and_then(|it| it.as_bool()) == Some(true);

            let snippet_text_edit = get_bool("snippetTextEdit");
            self.assist.allow_snippets(snippet_text_edit);

            self.client_caps.code_action_group = get_bool("codeActionGroup");
            self.client_caps.resolve_code_action = get_bool("resolveCodeAction");
            self.client_caps.hover_actions = get_bool("hoverActions");
            self.client_caps.status_notification = get_bool("statusNotification");
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ManifestOrProjectJson {
    Manifest(PathBuf),
    ProjectJson(ProjectJsonData),
}
