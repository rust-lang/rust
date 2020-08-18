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
use ide::{AssistConfig, CompletionConfig, DiagnosticsConfig, HoverConfig, InlayHintsConfig};
use lsp_types::ClientCapabilities;
use project_model::{CargoConfig, ProjectJson, ProjectJsonData, ProjectManifest};
use rustc_hash::FxHashSet;
use serde::Deserialize;
use vfs::AbsPathBuf;

use crate::diagnostics::DiagnosticsMapConfig;

#[derive(Debug, Clone)]
pub struct Config {
    pub client_caps: ClientCapsConfig,

    pub publish_diagnostics: bool,
    pub diagnostics: DiagnosticsConfig,
    pub diagnostics_map: DiagnosticsMapConfig,
    pub lru_capacity: Option<usize>,
    pub proc_macro_srv: Option<(PathBuf, Vec<OsString>)>,
    pub files: FilesConfig,
    pub notifications: NotificationsConfig,

    pub cargo_autoreload: bool,
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
    Rustfmt { extra_args: Vec<String> },
    CustomCommand { command: String, args: Vec<String> },
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
    pub signature_help_label_offsets: bool,
}

impl Config {
    pub fn new(root_path: AbsPathBuf) -> Self {
        Config {
            client_caps: ClientCapsConfig::default(),

            with_sysroot: true,
            publish_diagnostics: true,
            diagnostics: DiagnosticsConfig::default(),
            diagnostics_map: DiagnosticsMapConfig::default(),
            lru_capacity: None,
            proc_macro_srv: None,
            files: FilesConfig { watcher: FilesWatcher::Notify, exclude: Vec::new() },
            notifications: NotificationsConfig { cargo_toml_not_found: true },

            cargo_autoreload: true,
            cargo: CargoConfig::default(),
            rustfmt: RustfmtConfig::Rustfmt { extra_args: Vec::new() },
            flycheck: Some(FlycheckConfig::CargoCommand {
                command: "check".to_string(),
                target_triple: None,
                no_default_features: false,
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

    pub fn update(&mut self, json: serde_json::Value) {
        log::info!("Config::update({:#})", json);

        if json.is_null() || json.as_object().map_or(false, |it| it.is_empty()) {
            return;
        }

        let data = ConfigData::from_json(json);

        self.with_sysroot = data.withSysroot;
        self.publish_diagnostics = data.diagnostics_enable;
        self.diagnostics = DiagnosticsConfig {
            disable_experimental: !data.diagnostics_enableExperimental,
            disabled: data.diagnostics_disabled,
        };
        self.diagnostics_map = DiagnosticsMapConfig {
            warnings_as_info: data.diagnostics_warningsAsInfo,
            warnings_as_hint: data.diagnostics_warningsAsHint,
        };
        self.lru_capacity = data.lruCapacity;
        self.files.watcher = match data.files_watcher.as_str() {
            "notify" => FilesWatcher::Notify,
            "client" | _ => FilesWatcher::Client,
        };
        self.notifications =
            NotificationsConfig { cargo_toml_not_found: data.notifications_cargoTomlNotFound };
        self.cargo_autoreload = data.cargo_autoreload;
        self.cargo = CargoConfig {
            no_default_features: data.cargo_noDefaultFeatures,
            all_features: data.cargo_allFeatures,
            features: data.cargo_features.clone(),
            load_out_dirs_from_check: data.cargo_loadOutDirsFromCheck,
            target: data.cargo_target.clone(),
        };

        self.proc_macro_srv = if data.procMacro_enable {
            std::env::current_exe().ok().map(|path| (path, vec!["proc-macro".into()]))
        } else {
            None
        };

        self.rustfmt = match data.rustfmt_overrideCommand {
            Some(mut args) if !args.is_empty() => {
                let command = args.remove(0);
                RustfmtConfig::CustomCommand { command, args }
            }
            Some(_) | None => RustfmtConfig::Rustfmt { extra_args: data.rustfmt_extraArgs },
        };

        self.flycheck = if data.checkOnSave_enable {
            let flycheck_config = match data.checkOnSave_overrideCommand {
                Some(mut args) if !args.is_empty() => {
                    let command = args.remove(0);
                    FlycheckConfig::CustomCommand { command, args }
                }
                Some(_) | None => FlycheckConfig::CargoCommand {
                    command: data.checkOnSave_command,
                    target_triple: data.checkOnSave_target.or(data.cargo_target),
                    all_targets: data.checkOnSave_allTargets,
                    no_default_features: data
                        .checkOnSave_noDefaultFeatures
                        .unwrap_or(data.cargo_noDefaultFeatures),
                    all_features: data.checkOnSave_allFeatures.unwrap_or(data.cargo_allFeatures),
                    features: data.checkOnSave_features.unwrap_or(data.cargo_features),
                    extra_args: data.checkOnSave_extraArgs,
                },
            };
            Some(flycheck_config)
        } else {
            None
        };

        self.inlay_hints = InlayHintsConfig {
            type_hints: data.inlayHints_typeHints,
            parameter_hints: data.inlayHints_parameterHints,
            chaining_hints: data.inlayHints_chainingHints,
            max_length: data.inlayHints_maxLength,
        };

        self.completion.enable_postfix_completions = data.completion_postfix_enable;
        self.completion.add_call_parenthesis = data.completion_addCallParenthesis;
        self.completion.add_call_argument_snippets = data.completion_addCallArgumentSnippets;

        self.call_info_full = data.callInfo_full;

        self.lens = LensConfig {
            run: data.lens_enable && data.lens_run,
            debug: data.lens_enable && data.lens_debug,
            implementations: data.lens_enable && data.lens_implementations,
        };

        if !data.linkedProjects.is_empty() {
            self.linked_projects.clear();
            for linked_project in data.linkedProjects {
                let linked_project = match linked_project {
                    ManifestOrProjectJson::Manifest(it) => {
                        let path = self.root_path.join(it);
                        match ProjectManifest::from_manifest_file(path) {
                            Ok(it) => it.into(),
                            Err(_) => continue,
                        }
                    }
                    ManifestOrProjectJson::ProjectJson(it) => {
                        ProjectJson::new(&self.root_path, it).into()
                    }
                };
                self.linked_projects.push(linked_project);
            }
        }

        self.hover = HoverConfig {
            implementations: data.hoverActions_enable && data.hoverActions_implementations,
            run: data.hoverActions_enable && data.hoverActions_run,
            debug: data.hoverActions_enable && data.hoverActions_debug,
            goto_type_def: data.hoverActions_enable && data.hoverActions_gotoTypeDef,
        };

        log::info!("Config::update() = {:#?}", self);
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
            if let Some(value) = doc_caps
                .signature_help
                .as_ref()
                .and_then(|it| it.signature_information.as_ref())
                .and_then(|it| it.parameter_information.as_ref())
                .and_then(|it| it.label_offset_support)
            {
                self.client_caps.signature_help_label_offsets = value;
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

macro_rules! config_data {
    (struct $name:ident { $($field:ident: $ty:ty = $default:expr,)*}) => {
        #[allow(non_snake_case)]
        struct $name { $($field: $ty,)* }
        impl $name {
            fn from_json(mut json: serde_json::Value) -> $name {
                $name {$(
                    $field: {
                        let pointer = stringify!($field).replace('_', "/");
                        let pointer = format!("/{}", pointer);
                        json.pointer_mut(&pointer)
                            .and_then(|it| serde_json::from_value(it.take()).ok())
                            .unwrap_or($default)
                    },
                )*}
            }
        }

    };
}

config_data! {
    struct ConfigData {
        callInfo_full: bool = true,

        cargo_autoreload: bool           = true,
        cargo_allFeatures: bool          = false,
        cargo_features: Vec<String>      = Vec::new(),
        cargo_loadOutDirsFromCheck: bool = false,
        cargo_noDefaultFeatures: bool    = false,
        cargo_target: Option<String>     = None,

        checkOnSave_enable: bool                         = false,
        checkOnSave_allFeatures: Option<bool>            = None,
        checkOnSave_allTargets: bool                     = true,
        checkOnSave_command: String                      = "check".into(),
        checkOnSave_noDefaultFeatures: Option<bool>      = None,
        checkOnSave_target: Option<String>               = None,
        checkOnSave_extraArgs: Vec<String>               = Vec::new(),
        checkOnSave_features: Option<Vec<String>>        = None,
        checkOnSave_overrideCommand: Option<Vec<String>> = None,

        completion_addCallArgumentSnippets: bool = true,
        completion_addCallParenthesis: bool      = true,
        completion_postfix_enable: bool          = true,

        diagnostics_enable: bool                = true,
        diagnostics_enableExperimental: bool    = true,
        diagnostics_disabled: FxHashSet<String> = FxHashSet::default(),
        diagnostics_warningsAsHint: Vec<String> = Vec::new(),
        diagnostics_warningsAsInfo: Vec<String> = Vec::new(),

        files_watcher: String = "client".into(),

        hoverActions_debug: bool           = true,
        hoverActions_enable: bool          = true,
        hoverActions_gotoTypeDef: bool     = true,
        hoverActions_implementations: bool = true,
        hoverActions_run: bool             = true,

        inlayHints_chainingHints: bool      = true,
        inlayHints_maxLength: Option<usize> = None,
        inlayHints_parameterHints: bool     = true,
        inlayHints_typeHints: bool          = true,

        lens_debug: bool           = true,
        lens_enable: bool          = true,
        lens_implementations: bool = true,
        lens_run: bool             = true,

        linkedProjects: Vec<ManifestOrProjectJson> = Vec::new(),
        lruCapacity: Option<usize>                 = None,
        notifications_cargoTomlNotFound: bool      = true,
        procMacro_enable: bool                     = false,

        rustfmt_extraArgs: Vec<String>               = Vec::new(),
        rustfmt_overrideCommand: Option<Vec<String>> = None,

        withSysroot: bool = true,
    }
}
