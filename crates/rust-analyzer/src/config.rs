//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use std::{ffi::OsString, iter, path::PathBuf};

use flycheck::FlycheckConfig;
use hir::PrefixKind;
use ide::{AssistConfig, CompletionConfig, DiagnosticsConfig, HoverConfig, InlayHintsConfig};
use ide_db::helpers::{
    insert_use::{InsertUseConfig, MergeBehavior},
    SnippetCap,
};
use itertools::Itertools;
use lsp_types::{ClientCapabilities, MarkupKind};
use project_model::{CargoConfig, ProjectJson, ProjectJsonData, ProjectManifest, RustcSource};
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Deserialize};
use vfs::AbsPathBuf;

use crate::{
    caps::completion_item_edit_resolve, diagnostics::DiagnosticsMapConfig,
    line_index::OffsetEncoding, lsp_ext::supports_utf8,
};

config_data! {
    struct ConfigData {
        /// The strategy to use when inserting new imports or merging imports.
        assist_importMergeBehavior |
        assist_importMergeBehaviour: MergeBehaviorDef  = "\"full\"",
        /// The path structure for newly inserted paths to use.
        assist_importPrefix: ImportPrefixDef           = "\"plain\"",
        /// Group inserted imports by the [following order](https://rust-analyzer.github.io/manual.html#auto-import). Groups are separated by newlines.
        assist_importGroup: bool                       = "true",
        /// Show function name and docs in parameter hints.
        callInfo_full: bool = "true",

        /// Automatically refresh project info via `cargo metadata` on
        /// `Cargo.toml` changes.
        cargo_autoreload: bool           = "true",
        /// Activate all available features (`--all-features`).
        cargo_allFeatures: bool          = "false",
        /// List of features to activate.
        cargo_features: Vec<String>      = "[]",
        /// Run build scripts (`build.rs`) for more precise code analysis.
        cargo_runBuildScripts |
        cargo_loadOutDirsFromCheck: bool = "true",
        /// Do not activate the `default` feature.
        cargo_noDefaultFeatures: bool    = "false",
        /// Compilation target (target triple).
        cargo_target: Option<String>     = "null",
        /// Internal config for debugging, disables loading of sysroot crates.
        cargo_noSysroot: bool            = "false",

        /// Run specified `cargo check` command for diagnostics on save.
        checkOnSave_enable: bool                         = "true",
        /// Check with all features (`--all-features`).
        /// Defaults to `#rust-analyzer.cargo.allFeatures#`.
        checkOnSave_allFeatures: Option<bool>            = "null",
        /// Check all targets and tests (`--all-targets`).
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
        /// These directories will be ignored by rust-analyzer.
        files_excludeDirs: Vec<PathBuf> = "[]",

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
        /// Whether to show `References` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_references: bool = "false",

        /// Disable project auto-discovery in favor of explicitly specified set
        /// of projects.\n\nElements must be paths pointing to `Cargo.toml`,
        /// `rust-project.json`, or JSON objects in `rust-project.json` format.
        linkedProjects: Vec<ManifestOrProjectJson> = "[]",

        /// Number of syntax trees rust-analyzer keeps in memory. Defaults to 128.
        lruCapacity: Option<usize>                 = "null",

        /// Whether to show `can't find Cargo.toml` error message.
        notifications_cargoTomlNotFound: bool      = "true",

        /// Enable support for procedural macros, implies `#rust-analyzer.cargo.runBuildScripts#`.
        procMacro_enable: bool                     = "false",
        /// Internal config, path to proc-macro server executable (typically,
        /// this is rust-analyzer itself, but we override this in tests).
        procMacro_server: Option<PathBuf>          = "null",

        /// Command to be executed instead of 'cargo' for runnables.
        runnables_overrideCargo: Option<String> = "null",
        /// Additional arguments to be passed to cargo for runnables such as
        /// tests or binaries.\nFor example, it may be `--release`.
        runnables_cargoExtraArgs: Vec<String>   = "[]",

        /// Path to the rust compiler sources, for usage in rustc_private projects, or "discover"
        /// to try to automatically find it.
        rustcSource : Option<String> = "null",

        /// Additional arguments to `rustfmt`.
        rustfmt_extraArgs: Vec<String>               = "[]",
        /// Advanced option, fully override the command rust-analyzer uses for
        /// formatting.
        rustfmt_overrideCommand: Option<Vec<String>> = "null",
    }
}

impl Default for ConfigData {
    fn default() -> Self {
        ConfigData::from_json(serde_json::Value::Null)
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    caps: lsp_types::ClientCapabilities,
    data: ConfigData,
    pub discovered_projects: Option<Vec<ProjectManifest>>,
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
    pub refs: bool, // for Struct, Enum, Union and Trait
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
        self.method_refs || self.refs
    }
}

#[derive(Debug, Clone)]
pub struct FilesConfig {
    pub watcher: FilesWatcher,
    pub exclude: Vec<AbsPathBuf>,
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
#[derive(Debug, Clone)]
pub struct RunnablesConfig {
    /// Custom command to be executed instead of `cargo` for runnables.
    pub override_cargo: Option<String>,
    /// Additional arguments for the `cargo`, e.g. `--release`.
    pub cargo_extra_args: Vec<String>,
}

impl Config {
    pub fn new(root_path: AbsPathBuf, caps: ClientCapabilities) -> Self {
        Config { caps, data: ConfigData::default(), discovered_projects: None, root_path }
    }
    pub fn update(&mut self, json: serde_json::Value) {
        log::info!("updating config from JSON: {:#}", json);
        if json.is_null() || json.as_object().map_or(false, |it| it.is_empty()) {
            return;
        }
        self.data = ConfigData::from_json(json);
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
    pub fn linked_projects(&self) -> Vec<LinkedProject> {
        if self.data.linkedProjects.is_empty() {
            self.discovered_projects
                .as_ref()
                .into_iter()
                .flatten()
                .cloned()
                .map(LinkedProject::from)
                .collect()
        } else {
            self.data
                .linkedProjects
                .iter()
                .filter_map(|linked_project| {
                    let res = match linked_project {
                        ManifestOrProjectJson::Manifest(it) => {
                            let path = self.root_path.join(it);
                            ProjectManifest::from_manifest_file(path)
                                .map_err(|e| log::error!("failed to load linked project: {}", e))
                                .ok()?
                                .into()
                        }
                        ManifestOrProjectJson::ProjectJson(it) => {
                            ProjectJson::new(&self.root_path, it.clone()).into()
                        }
                    };
                    Some(res)
                })
                .collect()
        }
    }

    pub fn did_save_text_document_dynamic_registration(&self) -> bool {
        let caps =
            try_or!(self.caps.text_document.as_ref()?.synchronization.clone()?, Default::default());
        caps.did_save == Some(true) && caps.dynamic_registration == Some(true)
    }
    pub fn did_change_watched_files_dynamic_registration(&self) -> bool {
        try_or!(
            self.caps.workspace.as_ref()?.did_change_watched_files.as_ref()?.dynamic_registration?,
            false
        )
    }

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
    pub fn offset_encoding(&self) -> OffsetEncoding {
        if supports_utf8(&self.caps) {
            OffsetEncoding::Utf8
        } else {
            OffsetEncoding::Utf16
        }
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

    pub fn publish_diagnostics(&self) -> bool {
        self.data.diagnostics_enable
    }
    pub fn diagnostics(&self) -> DiagnosticsConfig {
        DiagnosticsConfig {
            disable_experimental: !self.data.diagnostics_enableExperimental,
            disabled: self.data.diagnostics_disabled.clone(),
        }
    }
    pub fn diagnostics_map(&self) -> DiagnosticsMapConfig {
        DiagnosticsMapConfig {
            warnings_as_info: self.data.diagnostics_warningsAsInfo.clone(),
            warnings_as_hint: self.data.diagnostics_warningsAsHint.clone(),
        }
    }
    pub fn lru_capacity(&self) -> Option<usize> {
        self.data.lruCapacity
    }
    pub fn proc_macro_srv(&self) -> Option<(PathBuf, Vec<OsString>)> {
        if !self.data.procMacro_enable {
            return None;
        }

        let path = self.data.procMacro_server.clone().or_else(|| std::env::current_exe().ok())?;
        Some((path, vec!["proc-macro".into()]))
    }
    pub fn files(&self) -> FilesConfig {
        FilesConfig {
            watcher: match self.data.files_watcher.as_str() {
                "notify" => FilesWatcher::Notify,
                "client" | _ => FilesWatcher::Client,
            },
            exclude: self.data.files_excludeDirs.iter().map(|it| self.root_path.join(it)).collect(),
        }
    }
    pub fn notifications(&self) -> NotificationsConfig {
        NotificationsConfig { cargo_toml_not_found: self.data.notifications_cargoTomlNotFound }
    }
    pub fn cargo_autoreload(&self) -> bool {
        self.data.cargo_autoreload
    }
    pub fn run_build_scripts(&self) -> bool {
        self.data.cargo_runBuildScripts || self.data.procMacro_enable
    }
    pub fn cargo(&self) -> CargoConfig {
        let rustc_source = self.data.rustcSource.as_ref().map(|rustc_src| {
            if rustc_src == "discover" {
                RustcSource::Discover
            } else {
                RustcSource::Path(self.root_path.join(rustc_src))
            }
        });

        CargoConfig {
            no_default_features: self.data.cargo_noDefaultFeatures,
            all_features: self.data.cargo_allFeatures,
            features: self.data.cargo_features.clone(),
            target: self.data.cargo_target.clone(),
            rustc_source,
            no_sysroot: self.data.cargo_noSysroot,
        }
    }
    pub fn rustfmt(&self) -> RustfmtConfig {
        match &self.data.rustfmt_overrideCommand {
            Some(args) if !args.is_empty() => {
                let mut args = args.clone();
                let command = args.remove(0);
                RustfmtConfig::CustomCommand { command, args }
            }
            Some(_) | None => {
                RustfmtConfig::Rustfmt { extra_args: self.data.rustfmt_extraArgs.clone() }
            }
        }
    }
    pub fn flycheck(&self) -> Option<FlycheckConfig> {
        if !self.data.checkOnSave_enable {
            return None;
        }
        let flycheck_config = match &self.data.checkOnSave_overrideCommand {
            Some(args) if !args.is_empty() => {
                let mut args = args.clone();
                let command = args.remove(0);
                FlycheckConfig::CustomCommand { command, args }
            }
            Some(_) | None => FlycheckConfig::CargoCommand {
                command: self.data.checkOnSave_command.clone(),
                target_triple: self
                    .data
                    .checkOnSave_target
                    .clone()
                    .or_else(|| self.data.cargo_target.clone()),
                all_targets: self.data.checkOnSave_allTargets,
                no_default_features: self
                    .data
                    .checkOnSave_noDefaultFeatures
                    .unwrap_or(self.data.cargo_noDefaultFeatures),
                all_features: self
                    .data
                    .checkOnSave_allFeatures
                    .unwrap_or(self.data.cargo_allFeatures),
                features: self
                    .data
                    .checkOnSave_features
                    .clone()
                    .unwrap_or_else(|| self.data.cargo_features.clone()),
                extra_args: self.data.checkOnSave_extraArgs.clone(),
            },
        };
        Some(flycheck_config)
    }
    pub fn runnables(&self) -> RunnablesConfig {
        RunnablesConfig {
            override_cargo: self.data.runnables_overrideCargo.clone(),
            cargo_extra_args: self.data.runnables_cargoExtraArgs.clone(),
        }
    }
    pub fn inlay_hints(&self) -> InlayHintsConfig {
        InlayHintsConfig {
            type_hints: self.data.inlayHints_typeHints,
            parameter_hints: self.data.inlayHints_parameterHints,
            chaining_hints: self.data.inlayHints_chainingHints,
            max_length: self.data.inlayHints_maxLength,
        }
    }
    fn insert_use_config(&self) -> InsertUseConfig {
        InsertUseConfig {
            merge: match self.data.assist_importMergeBehavior {
                MergeBehaviorDef::None => None,
                MergeBehaviorDef::Full => Some(MergeBehavior::Full),
                MergeBehaviorDef::Last => Some(MergeBehavior::Last),
            },
            prefix_kind: match self.data.assist_importPrefix {
                ImportPrefixDef::Plain => PrefixKind::Plain,
                ImportPrefixDef::ByCrate => PrefixKind::ByCrate,
                ImportPrefixDef::BySelf => PrefixKind::BySelf,
            },
            group: self.data.assist_importGroup,
        }
    }
    pub fn completion(&self) -> CompletionConfig {
        CompletionConfig {
            enable_postfix_completions: self.data.completion_postfix_enable,
            enable_imports_on_the_fly: self.data.completion_autoimport_enable
                && completion_item_edit_resolve(&self.caps),
            add_call_parenthesis: self.data.completion_addCallParenthesis,
            add_call_argument_snippets: self.data.completion_addCallArgumentSnippets,
            insert_use: self.insert_use_config(),
            snippet_cap: SnippetCap::new(try_or!(
                self.caps
                    .text_document
                    .as_ref()?
                    .completion
                    .as_ref()?
                    .completion_item
                    .as_ref()?
                    .snippet_support?,
                false
            )),
        }
    }
    pub fn assist(&self) -> AssistConfig {
        AssistConfig {
            snippet_cap: SnippetCap::new(self.experimental("snippetTextEdit")),
            allowed: None,
            insert_use: self.insert_use_config(),
        }
    }
    pub fn call_info_full(&self) -> bool {
        self.data.callInfo_full
    }
    pub fn lens(&self) -> LensConfig {
        LensConfig {
            run: self.data.lens_enable && self.data.lens_run,
            debug: self.data.lens_enable && self.data.lens_debug,
            implementations: self.data.lens_enable && self.data.lens_implementations,
            method_refs: self.data.lens_enable && self.data.lens_methodReferences,
            refs: self.data.lens_enable && self.data.lens_references,
        }
    }
    pub fn hover(&self) -> HoverConfig {
        HoverConfig {
            implementations: self.data.hoverActions_enable
                && self.data.hoverActions_implementations,
            run: self.data.hoverActions_enable && self.data.hoverActions_run,
            debug: self.data.hoverActions_enable && self.data.hoverActions_debug,
            goto_type_def: self.data.hoverActions_enable && self.data.hoverActions_gotoTypeDef,
            links_in_hover: self.data.hoverActions_linksInHover,
            markdown: try_or!(
                self.caps
                    .text_document
                    .as_ref()?
                    .hover
                    .as_ref()?
                    .content_format
                    .as_ref()?
                    .as_slice(),
                &[]
            )
            .contains(&MarkupKind::Markdown),
        }
    }
    pub fn semantic_tokens_refresh(&self) -> bool {
        try_or!(self.caps.workspace.as_ref()?.semantic_tokens.as_ref()?.refresh_support?, false)
    }
    pub fn code_lens_refresh(&self) -> bool {
        try_or!(self.caps.workspace.as_ref()?.code_lens.as_ref()?.refresh_support?, false)
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum ManifestOrProjectJson {
    Manifest(PathBuf),
    ProjectJson(ProjectJsonData),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum MergeBehaviorDef {
    None,
    Full,
    Last,
}

#[derive(Deserialize, Debug, Clone)]
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
            $field:ident $(| $alias:ident)?: $ty:ty = $default:expr,
        )*
    }) => {
        #[allow(non_snake_case)]
        #[derive(Debug, Clone)]
        struct $name { $($field: $ty,)* }
        impl $name {
            fn from_json(mut json: serde_json::Value) -> $name {
                $name {$(
                    $field: get_field(
                        &mut json,
                        stringify!($field),
                        None$(.or(Some(stringify!($alias))))?,
                        $default,
                    ),
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
    alias: Option<&'static str>,
    default: &str,
) -> T {
    let default = serde_json::from_str(default).unwrap();

    // XXX: check alias first, to work-around the VS Code where it pre-fills the
    // defaults instead of sending an empty object.
    alias
        .into_iter()
        .chain(iter::once(field))
        .find_map(move |field| {
            let mut pointer = field.replace('_', "/");
            pointer.insert(0, '/');
            json.pointer_mut(&pointer).and_then(|it| serde_json::from_value(it.take()).ok())
        })
        .unwrap_or(default)
}

fn schema(fields: &[(&'static str, &'static str, &[&str], &str)]) -> serde_json::Value {
    for ((f1, ..), (f2, ..)) in fields.iter().zip(&fields[1..]) {
        fn key(f: &str) -> &str {
            f.splitn(2, '_').next().unwrap()
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
        "Vec<PathBuf>" => set! {
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
        "Option<PathBuf>" => set! {
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
        let mut schema = schema
            .trim_start_matches('{')
            .trim_end_matches('}')
            .replace("  ", "    ")
            .replace("\n", "\n            ")
            .trim_start_matches('\n')
            .trim_end()
            .to_string();
        schema.push_str(",\n");

        let package_json_path = project_dir().join("editors/code/package.json");
        let mut package_json = fs::read_to_string(&package_json_path).unwrap();

        let start_marker = "                \"$generated-start\": false,\n";
        let end_marker = "                \"$generated-end\": false\n";

        let start = package_json.find(start_marker).unwrap() + start_marker.len();
        let end = package_json.find(end_marker).unwrap();
        let p = remove_ws(&package_json[start..end]);
        let s = remove_ws(&schema);

        if !p.contains(&s) {
            package_json.replace_range(start..end, &schema);
            fs::write(&package_json_path, &mut package_json).unwrap();
            panic!("new config, updating package.json")
        }
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
