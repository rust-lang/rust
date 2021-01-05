//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use std::{convert::TryFrom, ffi::OsString, path::PathBuf};

use flycheck::FlycheckConfig;
use hir::PrefixKind;
use ide::{AssistConfig, CompletionConfig, DiagnosticsConfig, HoverConfig, InlayHintsConfig};
use ide_db::helpers::insert_use::MergeBehavior;
use itertools::Itertools;
use lsp_types::{ClientCapabilities, MarkupKind};
use project_model::{CargoConfig, ProjectJson, ProjectJsonData, ProjectManifest};
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Deserialize};
use vfs::AbsPathBuf;

use crate::{caps::enabled_completions_resolve_capabilities, diagnostics::DiagnosticsMapConfig};

config_data! {
    struct ConfigData {
        /// The strategy to use when inserting new imports or merging imports.
        assist_importMergeBehaviour: MergeBehaviorDef = "\"full\"",
        /// The path structure for newly inserted paths to use.
        assist_importPrefix: ImportPrefixDef           = "\"plain\"",

        /// Show function name and docs in parameter hints.
        callInfo_full: bool = "true",

        /// Automatically refresh project info via `cargo metadata` on
        /// `Cargo.toml` changes.
        cargo_autoreload: bool           = "true",
        /// Activate all available features.
        cargo_allFeatures: bool          = "false",
        /// List of features to activate.
        cargo_features: Vec<String>      = "[]",
        /// Run `cargo check` on startup to get the correct value for package
        /// OUT_DIRs.
        cargo_loadOutDirsFromCheck: bool = "false",
        /// Do not activate the `default` feature.
        cargo_noDefaultFeatures: bool    = "false",
        /// Compilation target (target triple).
        cargo_target: Option<String>     = "null",
        /// Internal config for debugging, disables loading of sysroot crates.
        cargo_noSysroot: bool            = "false",

        /// Run specified `cargo check` command for diagnostics on save.
        checkOnSave_enable: bool                         = "true",
        /// Check with all features (will be passed as `--all-features`).
        /// Defaults to `#rust-analyzer.cargo.allFeatures#`.
        checkOnSave_allFeatures: Option<bool>            = "null",
        /// Check all targets and tests (will be passed as `--all-targets`).
        checkOnSave_allTargets: bool                     = "true",
        /// Cargo command to use for `cargo check`.
        checkOnSave_command: String                      = "\"check\"",
        /// Do not activate the `default` feature.
        checkOnSave_noDefaultFeatures: Option<bool>      = "null",
        /// Check for a specific target. Defaults to
        /// `#rust-analyzer.cargo.target#`.
        checkOnSave_target: Option<String>               = "null",
        /// Extra arguments for `cargo check`.
        checkOnSave_extraArgs: Vec<String>               = "[]",
        /// List of features to activate. Defaults to
        /// `#rust-analyzer.cargo.features#`.
        checkOnSave_features: Option<Vec<String>>        = "null",
        /// Advanced option, fully override the command rust-analyzer uses for
        /// checking. The command should include `--message-format=json` or
        /// similar option.
        checkOnSave_overrideCommand: Option<Vec<String>> = "null",

        /// Whether to add argument snippets when completing functions.
        completion_addCallArgumentSnippets: bool = "true",
        /// Whether to add parenthesis when completing functions.
        completion_addCallParenthesis: bool      = "true",
        /// Whether to show postfix snippets like `dbg`, `if`, `not`, etc.
        completion_postfix_enable: bool          = "true",
        /// Toggles the additional completions that automatically add imports when completed.
        /// Note that your client must specify the `additionalTextEdits` LSP client capability to truly have this feature enabled.
        completion_autoimport_enable: bool       = "true",

        /// Whether to show native rust-analyzer diagnostics.
        diagnostics_enable: bool                = "true",
        /// Whether to show experimental rust-analyzer diagnostics that might
        /// have more false positives than usual.
        diagnostics_enableExperimental: bool    = "true",
        /// List of rust-analyzer diagnostics to disable.
        diagnostics_disabled: FxHashSet<String> = "[]",
        /// List of warnings that should be displayed with info severity.\n\nThe
        /// warnings will be indicated by a blue squiggly underline in code and
        /// a blue icon in the `Problems Panel`.
        diagnostics_warningsAsHint: Vec<String> = "[]",
        /// List of warnings that should be displayed with hint severity.\n\nThe
        /// warnings will be indicated by faded text or three dots in code and
        /// will not show up in the `Problems Panel`.
        diagnostics_warningsAsInfo: Vec<String> = "[]",

        /// Controls file watching implementation.
        files_watcher: String = "\"client\"",

        /// Whether to show `Debug` action. Only applies when
        /// `#rust-analyzer.hoverActions.enable#` is set.
        hoverActions_debug: bool           = "true",
        /// Whether to show HoverActions in Rust files.
        hoverActions_enable: bool          = "true",
        /// Whether to show `Go to Type Definition` action. Only applies when
        /// `#rust-analyzer.hoverActions.enable#` is set.
        hoverActions_gotoTypeDef: bool     = "true",
        /// Whether to show `Implementations` action. Only applies when
        /// `#rust-analyzer.hoverActions.enable#` is set.
        hoverActions_implementations: bool = "true",
        /// Whether to show `Run` action. Only applies when
        /// `#rust-analyzer.hoverActions.enable#` is set.
        hoverActions_run: bool             = "true",
        /// Use markdown syntax for links in hover.
        hoverActions_linksInHover: bool    = "true",

        /// Whether to show inlay type hints for method chains.
        inlayHints_chainingHints: bool      = "true",
        /// Maximum length for inlay hints. Default is unlimited.
        inlayHints_maxLength: Option<usize> = "null",
        /// Whether to show function parameter name inlay hints at the call
        /// site.
        inlayHints_parameterHints: bool     = "true",
        /// Whether to show inlay type hints for variables.
        inlayHints_typeHints: bool          = "true",

        /// Whether to show `Debug` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_debug: bool            = "true",
        /// Whether to show CodeLens in Rust files.
        lens_enable: bool           = "true",
        /// Whether to show `Implementations` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_implementations: bool  = "true",
        /// Whether to show `Run` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_run: bool              = "true",
        /// Whether to show `Method References` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_methodReferences: bool = "false",

        /// Disable project auto-discovery in favor of explicitly specified set
        /// of projects.\n\nElements must be paths pointing to `Cargo.toml`,
        /// `rust-project.json`, or JSON objects in `rust-project.json` format.
        linkedProjects: Vec<ManifestOrProjectJson> = "[]",
        /// Number of syntax trees rust-analyzer keeps in memory.  Defaults to 128.
        lruCapacity: Option<usize>                 = "null",
        /// Whether to show `can't find Cargo.toml` error message.
        notifications_cargoTomlNotFound: bool      = "true",
        /// Enable Proc macro support, `#rust-analyzer.cargo.loadOutDirsFromCheck#` must be
        /// enabled.
        procMacro_enable: bool                     = "false",

        /// Command to be executed instead of 'cargo' for runnables.
        runnables_overrideCargo: Option<String> = "null",
        /// Additional arguments to be passed to cargo for runnables such as
        /// tests or binaries.\nFor example, it may be `--release`.
        runnables_cargoExtraArgs: Vec<String>   = "[]",

        /// Path to the rust compiler sources, for usage in rustc_private projects.
        rustcSource : Option<String> = "null",

        /// Additional arguments to `rustfmt`.
        rustfmt_extraArgs: Vec<String>               = "[]",
        /// Advanced option, fully override the command rust-analyzer uses for
        /// formatting.
        rustfmt_overrideCommand: Option<Vec<String>> = "null",
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub caps: lsp_types::ClientCapabilities,

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
    pub runnables: RunnablesConfig,

    pub inlay_hints: InlayHintsConfig,
    pub completion: CompletionConfig,
    pub assist: AssistConfig,
    pub call_info_full: bool,
    pub lens: LensConfig,
    pub hover: HoverConfig,
    pub semantic_tokens_refresh: bool,
    pub code_lens_refresh: bool,

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
    pub method_refs: bool,
}

impl Default for LensConfig {
    fn default() -> Self {
        Self { run: true, debug: true, implementations: true, method_refs: false }
    }
}

impl LensConfig {
    pub fn any(&self) -> bool {
        self.implementations || self.runnable() || self.references()
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn runnable(&self) -> bool {
        self.run || self.debug
    }

    pub fn references(&self) -> bool {
        self.method_refs
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

/// Configuration for runnable items, such as `main` function or tests.
#[derive(Debug, Clone, Default)]
pub struct RunnablesConfig {
    /// Custom command to be executed instead of `cargo` for runnables.
    pub override_cargo: Option<String>,
    /// Additional arguments for the `cargo`, e.g. `--release`.
    pub cargo_extra_args: Vec<String>,
}

impl Config {
    pub fn new(root_path: AbsPathBuf) -> Self {
        // Defaults here don't matter, we'll immediately re-write them with
        // ConfigData.
        let mut res = Config {
            caps: lsp_types::ClientCapabilities::default(),

            publish_diagnostics: false,
            diagnostics: DiagnosticsConfig::default(),
            diagnostics_map: DiagnosticsMapConfig::default(),
            lru_capacity: None,
            proc_macro_srv: None,
            files: FilesConfig { watcher: FilesWatcher::Notify, exclude: Vec::new() },
            notifications: NotificationsConfig { cargo_toml_not_found: false },

            cargo_autoreload: false,
            cargo: CargoConfig::default(),
            rustfmt: RustfmtConfig::Rustfmt { extra_args: Vec::new() },
            flycheck: Some(FlycheckConfig::CargoCommand {
                command: String::new(),
                target_triple: None,
                no_default_features: false,
                all_targets: false,
                all_features: false,
                extra_args: Vec::new(),
                features: Vec::new(),
            }),
            runnables: RunnablesConfig::default(),

            inlay_hints: InlayHintsConfig {
                type_hints: false,
                parameter_hints: false,
                chaining_hints: false,
                max_length: None,
            },
            completion: CompletionConfig::default(),
            assist: AssistConfig::default(),
            call_info_full: false,
            lens: LensConfig::default(),
            hover: HoverConfig::default(),
            semantic_tokens_refresh: false,
            code_lens_refresh: false,
            linked_projects: Vec::new(),
            root_path,
        };
        res.do_update(serde_json::json!({}));
        res
    }
    pub fn update(&mut self, json: serde_json::Value) {
        log::info!("updating config from JSON: {:#}", json);
        if json.is_null() || json.as_object().map_or(false, |it| it.is_empty()) {
            return;
        }
        self.do_update(json);
        log::info!("updated config: {:#?}", self);
    }
    fn do_update(&mut self, json: serde_json::Value) {
        let data = ConfigData::from_json(json);

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

        let rustc_source = if let Some(rustc_source) = data.rustcSource {
            let rustpath: PathBuf = rustc_source.into();
            AbsPathBuf::try_from(rustpath)
                .map_err(|_| {
                    log::error!("rustc source directory must be an absolute path");
                })
                .ok()
        } else {
            None
        };

        self.cargo = CargoConfig {
            no_default_features: data.cargo_noDefaultFeatures,
            all_features: data.cargo_allFeatures,
            features: data.cargo_features.clone(),
            load_out_dirs_from_check: data.cargo_loadOutDirsFromCheck,
            target: data.cargo_target.clone(),
            rustc_source: rustc_source,
            no_sysroot: data.cargo_noSysroot,
        };
        self.runnables = RunnablesConfig {
            override_cargo: data.runnables_overrideCargo,
            cargo_extra_args: data.runnables_cargoExtraArgs,
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

        self.assist.insert_use.merge = match data.assist_importMergeBehaviour {
            MergeBehaviorDef::None => None,
            MergeBehaviorDef::Full => Some(MergeBehavior::Full),
            MergeBehaviorDef::Last => Some(MergeBehavior::Last),
        };
        self.assist.insert_use.prefix_kind = match data.assist_importPrefix {
            ImportPrefixDef::Plain => PrefixKind::Plain,
            ImportPrefixDef::ByCrate => PrefixKind::ByCrate,
            ImportPrefixDef::BySelf => PrefixKind::BySelf,
        };

        self.completion.enable_postfix_completions = data.completion_postfix_enable;
        self.completion.enable_autoimport_completions = data.completion_autoimport_enable;
        self.completion.add_call_parenthesis = data.completion_addCallParenthesis;
        self.completion.add_call_argument_snippets = data.completion_addCallArgumentSnippets;
        self.completion.merge = self.assist.insert_use.merge;

        self.call_info_full = data.callInfo_full;

        self.lens = LensConfig {
            run: data.lens_enable && data.lens_run,
            debug: data.lens_enable && data.lens_debug,
            implementations: data.lens_enable && data.lens_implementations,
            method_refs: data.lens_enable && data.lens_methodReferences,
        };

        if !data.linkedProjects.is_empty() {
            self.linked_projects.clear();
            for linked_project in data.linkedProjects {
                let linked_project = match linked_project {
                    ManifestOrProjectJson::Manifest(it) => {
                        let path = self.root_path.join(it);
                        match ProjectManifest::from_manifest_file(path) {
                            Ok(it) => it.into(),
                            Err(e) => {
                                log::error!("failed to load linked project: {}", e);
                                continue;
                            }
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
            links_in_hover: data.hoverActions_linksInHover,
            markdown: true,
        };
    }

    pub fn update_caps(&mut self, caps: &ClientCapabilities) {
        self.caps = caps.clone();
        if let Some(doc_caps) = caps.text_document.as_ref() {
            if let Some(value) = doc_caps.hover.as_ref().and_then(|it| it.content_format.as_ref()) {
                self.hover.markdown = value.contains(&MarkupKind::Markdown)
            }

            self.completion.allow_snippets(false);
            self.completion.active_resolve_capabilities =
                enabled_completions_resolve_capabilities(caps).unwrap_or_default();
            if let Some(completion) = &doc_caps.completion {
                if let Some(completion_item) = &completion.completion_item {
                    if let Some(value) = completion_item.snippet_support {
                        self.completion.allow_snippets(value);
                    }
                }
            }
        }

        self.assist.allow_snippets(false);
        if let Some(experimental) = &caps.experimental {
            let get_bool =
                |index: &str| experimental.get(index).and_then(|it| it.as_bool()) == Some(true);

            let snippet_text_edit = get_bool("snippetTextEdit");
            self.assist.allow_snippets(snippet_text_edit);
        }

        if let Some(workspace_caps) = caps.workspace.as_ref() {
            if let Some(refresh_support) =
                workspace_caps.semantic_tokens.as_ref().and_then(|it| it.refresh_support)
            {
                self.semantic_tokens_refresh = refresh_support;
            }

            if let Some(refresh_support) =
                workspace_caps.code_lens.as_ref().and_then(|it| it.refresh_support)
            {
                self.code_lens_refresh = refresh_support;
            }
        }
    }

    pub fn json_schema() -> serde_json::Value {
        ConfigData::json_schema()
    }
}

macro_rules! try_ {
    ($expr:expr) => {
        || -> _ { Some($expr) }()
    };
}
macro_rules! try_or {
    ($expr:expr, $or:expr) => {
        try_!($expr).unwrap_or($or)
    };
}

impl Config {
    pub fn location_link(&self) -> bool {
        try_or!(self.caps.text_document.as_ref()?.definition?.link_support?, false)
    }
    pub fn line_folding_only(&self) -> bool {
        try_or!(self.caps.text_document.as_ref()?.folding_range.as_ref()?.line_folding_only?, false)
    }
    pub fn hierarchical_symbols(&self) -> bool {
        try_or!(
            self.caps
                .text_document
                .as_ref()?
                .document_symbol
                .as_ref()?
                .hierarchical_document_symbol_support?,
            false
        )
    }
    pub fn code_action_literals(&self) -> bool {
        try_!(self
            .caps
            .text_document
            .as_ref()?
            .code_action
            .as_ref()?
            .code_action_literal_support
            .as_ref()?)
        .is_some()
    }
    pub fn work_done_progress(&self) -> bool {
        try_or!(self.caps.window.as_ref()?.work_done_progress?, false)
    }
    pub fn code_action_resolve(&self) -> bool {
        try_or!(
            self.caps
                .text_document
                .as_ref()?
                .code_action
                .as_ref()?
                .resolve_support
                .as_ref()?
                .properties
                .as_slice(),
            &[]
        )
        .iter()
        .any(|it| it == "edit")
    }
    pub fn signature_help_label_offsets(&self) -> bool {
        try_or!(
            self.caps
                .text_document
                .as_ref()?
                .signature_help
                .as_ref()?
                .signature_information
                .as_ref()?
                .parameter_information
                .as_ref()?
                .label_offset_support?,
            false
        )
    }

    fn experimental(&self, index: &'static str) -> bool {
        try_or!(self.caps.experimental.as_ref()?.get(index)?.as_bool()?, false)
    }
    pub fn code_action_group(&self) -> bool {
        self.experimental("codeActionGroup")
    }
    pub fn hover_actions(&self) -> bool {
        self.experimental("hoverActions")
    }
    pub fn status_notification(&self) -> bool {
        self.experimental("statusNotification")
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ManifestOrProjectJson {
    Manifest(PathBuf),
    ProjectJson(ProjectJsonData),
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum MergeBehaviorDef {
    None,
    Full,
    Last,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ImportPrefixDef {
    Plain,
    BySelf,
    ByCrate,
}

macro_rules! _config_data {
    (struct $name:ident {
        $(
            $(#[doc=$doc:literal])*
            $field:ident: $ty:ty = $default:expr,
        )*
    }) => {
        #[allow(non_snake_case)]
        struct $name { $($field: $ty,)* }
        impl $name {
            fn from_json(mut json: serde_json::Value) -> $name {
                $name {$(
                    $field: get_field(&mut json, stringify!($field), $default),
                )*}
            }

            fn json_schema() -> serde_json::Value {
                schema(&[
                    $({
                        let field = stringify!($field);
                        let ty = stringify!($ty);
                        (field, ty, &[$($doc),*], $default)
                    },)*
                ])
            }

            #[cfg(test)]
            fn manual() -> String {
                manual(&[
                    $({
                        let field = stringify!($field);
                        let ty = stringify!($ty);
                        (field, ty, &[$($doc),*], $default)
                    },)*
                ])
            }
        }
    };
}
use _config_data as config_data;

fn get_field<T: DeserializeOwned>(
    json: &mut serde_json::Value,
    field: &'static str,
    default: &str,
) -> T {
    let default = serde_json::from_str(default).unwrap();

    let mut pointer = field.replace('_', "/");
    pointer.insert(0, '/');
    json.pointer_mut(&pointer)
        .and_then(|it| serde_json::from_value(it.take()).ok())
        .unwrap_or(default)
}

fn schema(fields: &[(&'static str, &'static str, &[&str], &str)]) -> serde_json::Value {
    for ((f1, ..), (f2, ..)) in fields.iter().zip(&fields[1..]) {
        fn key(f: &str) -> &str {
            f.splitn(2, "_").next().unwrap()
        }
        assert!(key(f1) <= key(f2), "wrong field order: {:?} {:?}", f1, f2);
    }

    let map = fields
        .iter()
        .map(|(field, ty, doc, default)| {
            let name = field.replace("_", ".");
            let name = format!("rust-analyzer.{}", name);
            let props = field_props(field, ty, doc, default);
            (name, props)
        })
        .collect::<serde_json::Map<_, _>>();
    map.into()
}

fn field_props(field: &str, ty: &str, doc: &[&str], default: &str) -> serde_json::Value {
    let doc = doc.iter().map(|it| it.trim()).join(" ");
    assert!(
        doc.ends_with('.') && doc.starts_with(char::is_uppercase),
        "bad docs for {}: {:?}",
        field,
        doc
    );
    let default = default.parse::<serde_json::Value>().unwrap();

    let mut map = serde_json::Map::default();
    macro_rules! set {
        ($($key:literal: $value:tt),*$(,)?) => {{$(
            map.insert($key.into(), serde_json::json!($value));
        )*}};
    }
    set!("markdownDescription": doc);
    set!("default": default);

    match ty {
        "bool" => set!("type": "boolean"),
        "String" => set!("type": "string"),
        "Vec<String>" => set! {
            "type": "array",
            "items": { "type": "string" },
        },
        "FxHashSet<String>" => set! {
            "type": "array",
            "items": { "type": "string" },
            "uniqueItems": true,
        },
        "Option<usize>" => set! {
            "type": ["null", "integer"],
            "minimum": 0,
        },
        "Option<String>" => set! {
            "type": ["null", "string"],
        },
        "Option<bool>" => set! {
            "type": ["null", "boolean"],
        },
        "Option<Vec<String>>" => set! {
            "type": ["null", "array"],
            "items": { "type": "string" },
        },
        "MergeBehaviorDef" => set! {
            "type": "string",
            "enum": ["none", "full", "last"],
            "enumDescriptions": [
                "No merging",
                "Merge all layers of the import trees",
                "Only merge the last layer of the import trees"
            ],
        },
        "ImportPrefixDef" => set! {
            "type": "string",
            "enum": [
                "plain",
                "by_self",
                "by_crate"
            ],
            "enumDescriptions": [
                "Insert import paths relative to the current module, using up to one `super` prefix if the parent module contains the requested item.",
                "Prefix all import paths with `self` if they don't begin with `self`, `super`, `crate` or a crate name.",
                "Force import paths to be absolute by always starting them with `crate` or the crate name they refer to."
            ],
        },
        "Vec<ManifestOrProjectJson>" => set! {
            "type": "array",
            "items": { "type": ["string", "object"] },
        },
        _ => panic!("{}: {}", ty, default),
    }

    map.into()
}

#[cfg(test)]
fn manual(fields: &[(&'static str, &'static str, &[&str], &str)]) -> String {
    fields
        .iter()
        .map(|(field, _ty, doc, default)| {
            let name = format!("rust-analyzer.{}", field.replace("_", "."));
            format!("[[{}]]{} (default: `{}`)::\n{}\n", name, name, default, doc.join(" "))
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use test_utils::project_dir;

    use super::*;

    #[test]
    fn schema_in_sync_with_package_json() {
        let s = Config::json_schema();
        let schema = format!("{:#}", s);
        let schema = schema.trim_start_matches('{').trim_end_matches('}');

        let package_json = project_dir().join("editors/code/package.json");
        let package_json = fs::read_to_string(&package_json).unwrap();

        let p = remove_ws(&package_json);
        let s = remove_ws(&schema);

        assert!(p.contains(&s), "update config in package.json. New config:\n{:#}", schema);
    }

    #[test]
    fn schema_in_sync_with_docs() {
        let docs_path = project_dir().join("docs/user/generated_config.adoc");
        let current = fs::read_to_string(&docs_path).unwrap();
        let expected = ConfigData::manual();

        if remove_ws(&current) != remove_ws(&expected) {
            fs::write(&docs_path, expected).unwrap();
            panic!("updated config manual");
        }
    }

    fn remove_ws(text: &str) -> String {
        text.replace(char::is_whitespace, "")
    }
}
