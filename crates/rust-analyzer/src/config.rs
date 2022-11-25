//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use std::{ffi::OsString, fmt, iter, path::PathBuf};

use flycheck::FlycheckConfig;
use ide::{
    AssistConfig, CallableSnippets, CompletionConfig, DiagnosticsConfig, ExprFillDefaultMode,
    HighlightRelatedConfig, HoverConfig, HoverDocFormat, InlayHintsConfig, JoinLinesConfig,
    Snippet, SnippetScope,
};
use ide_db::{
    imports::insert_use::{ImportGranularity, InsertUseConfig, PrefixKind},
    SnippetCap,
};
use itertools::Itertools;
use lsp_types::{ClientCapabilities, MarkupKind};
use project_model::{
    CargoConfig, ProjectJson, ProjectJsonData, ProjectManifest, RustcSource, UnsetTestCrates,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{de::DeserializeOwned, Deserialize};
use vfs::AbsPathBuf;

use crate::{
    caps::completion_item_edit_resolve,
    diagnostics::DiagnosticsMapConfig,
    line_index::OffsetEncoding,
    lsp_ext::{self, supports_utf8, WorkspaceSymbolSearchKind, WorkspaceSymbolSearchScope},
};

mod patch_old_style;

// Conventions for configuration keys to preserve maximal extendability without breakage:
//  - Toggles (be it binary true/false or with more options in-between) should almost always suffix as `_enable`
//    This has the benefit of namespaces being extensible, and if the suffix doesn't fit later it can be changed without breakage.
//  - In general be wary of using the namespace of something verbatim, it prevents us from adding subkeys in the future
//  - Don't use abbreviations unless really necessary
//  - foo_command = overrides the subcommand, foo_overrideCommand allows full overwriting, extra args only applies for foo_command

// Defines the server-side configuration of the rust-analyzer. We generate
// *parts* of VS Code's `package.json` config from this. Run `cargo test` to
// re-generate that file.
//
// However, editor specific config, which the server doesn't know about, should
// be specified directly in `package.json`.
//
// To deprecate an option by replacing it with another name use `new_name | `old_name` so that we keep
// parsing the old name.
config_data! {
    struct ConfigData {
        /// Placeholder expression to use for missing expressions in assists.
        assist_expressionFillDefault: ExprFillDefaultDef              = "\"todo\"",

        /// Warm up caches on project load.
        cachePriming_enable: bool = "true",
        /// How many worker threads to handle priming caches. The default `0` means to pick automatically.
        cachePriming_numThreads: ParallelCachePrimingNumThreads = "0",

        /// Automatically refresh project info via `cargo metadata` on
        /// `Cargo.toml` or `.cargo/config.toml` changes.
        cargo_autoreload: bool           = "true",
        /// Run build scripts (`build.rs`) for more precise code analysis.
        cargo_buildScripts_enable: bool  = "true",
        /// Override the command rust-analyzer uses to run build scripts and
        /// build procedural macros. The command is required to output json
        /// and should therefore include `--message-format=json` or a similar
        /// option.
        ///
        /// By default, a cargo invocation will be constructed for the configured
        /// targets and features, with the following base command line:
        ///
        /// ```bash
        /// cargo check --quiet --workspace --message-format=json --all-targets
        /// ```
        /// .
        cargo_buildScripts_overrideCommand: Option<Vec<String>> = "null",
        /// Use `RUSTC_WRAPPER=rust-analyzer` when running build scripts to
        /// avoid checking unnecessary things.
        cargo_buildScripts_useRustcWrapper: bool = "true",
        /// List of features to activate.
        ///
        /// Set this to `"all"` to pass `--all-features` to cargo.
        cargo_features: CargoFeatures      = "[]",
        /// Whether to pass `--no-default-features` to cargo.
        cargo_noDefaultFeatures: bool    = "false",
        /// Internal config for debugging, disables loading of sysroot crates.
        cargo_noSysroot: bool            = "false",
        /// Compilation target override (target triple).
        cargo_target: Option<String>     = "null",
        /// Unsets `#[cfg(test)]` for the specified crates.
        cargo_unsetTest: Vec<String>   = "[\"core\"]",

        /// Check all targets and tests (`--all-targets`).
        checkOnSave_allTargets: bool                     = "true",
        /// Cargo command to use for `cargo check`.
        checkOnSave_command: String                      = "\"check\"",
        /// Run specified `cargo check` command for diagnostics on save.
        checkOnSave_enable: bool                         = "true",
        /// Extra arguments for `cargo check`.
        checkOnSave_extraArgs: Vec<String>               = "[]",
        /// List of features to activate. Defaults to
        /// `#rust-analyzer.cargo.features#`.
        ///
        /// Set to `"all"` to pass `--all-features` to Cargo.
        checkOnSave_features: Option<CargoFeatures>      = "null",
        /// Whether to pass `--no-default-features` to Cargo. Defaults to
        /// `#rust-analyzer.cargo.noDefaultFeatures#`.
        checkOnSave_noDefaultFeatures: Option<bool>      = "null",
        /// Override the command rust-analyzer uses instead of `cargo check` for
        /// diagnostics on save. The command is required to output json and
        /// should therefor include `--message-format=json` or a similar option.
        ///
        /// If you're changing this because you're using some tool wrapping
        /// Cargo, you might also want to change
        /// `#rust-analyzer.cargo.buildScripts.overrideCommand#`.
        ///
        /// If there are multiple linked projects, this command is invoked for
        /// each of them, with the working directory being the project root
        /// (i.e., the folder containing the `Cargo.toml`).
        ///
        /// An example command would be:
        ///
        /// ```bash
        /// cargo check --workspace --message-format=json --all-targets
        /// ```
        /// .
        checkOnSave_overrideCommand: Option<Vec<String>> = "null",
        /// Check for a specific target. Defaults to
        /// `#rust-analyzer.cargo.target#`.
        checkOnSave_target: Option<String>               = "null",

        /// Toggles the additional completions that automatically add imports when completed.
        /// Note that your client must specify the `additionalTextEdits` LSP client capability to truly have this feature enabled.
        completion_autoimport_enable: bool       = "true",
        /// Toggles the additional completions that automatically show method calls and field accesses
        /// with `self` prefixed to them when inside a method.
        completion_autoself_enable: bool        = "true",
        /// Whether to add parenthesis and argument snippets when completing function.
        completion_callable_snippets: CallableCompletionDef  = "\"fill_arguments\"",
        /// Whether to show postfix snippets like `dbg`, `if`, `not`, etc.
        completion_postfix_enable: bool         = "true",
        /// Enables completions of private items and fields that are defined in the current workspace even if they are not visible at the current position.
        completion_privateEditable_enable: bool = "false",
        /// Custom completion snippets.
        // NOTE: Keep this list in sync with the feature docs of user snippets.
        completion_snippets_custom: FxHashMap<String, SnippetDef> = r#"{
            "Arc::new": {
                "postfix": "arc",
                "body": "Arc::new(${receiver})",
                "requires": "std::sync::Arc",
                "description": "Put the expression into an `Arc`",
                "scope": "expr"
            },
            "Rc::new": {
                "postfix": "rc",
                "body": "Rc::new(${receiver})",
                "requires": "std::rc::Rc",
                "description": "Put the expression into an `Rc`",
                "scope": "expr"
            },
            "Box::pin": {
                "postfix": "pinbox",
                "body": "Box::pin(${receiver})",
                "requires": "std::boxed::Box",
                "description": "Put the expression into a pinned `Box`",
                "scope": "expr"
            },
            "Ok": {
                "postfix": "ok",
                "body": "Ok(${receiver})",
                "description": "Wrap the expression in a `Result::Ok`",
                "scope": "expr"
            },
            "Err": {
                "postfix": "err",
                "body": "Err(${receiver})",
                "description": "Wrap the expression in a `Result::Err`",
                "scope": "expr"
            },
            "Some": {
                "postfix": "some",
                "body": "Some(${receiver})",
                "description": "Wrap the expression in an `Option::Some`",
                "scope": "expr"
            }
        }"#,

        /// List of rust-analyzer diagnostics to disable.
        diagnostics_disabled: FxHashSet<String> = "[]",
        /// Whether to show native rust-analyzer diagnostics.
        diagnostics_enable: bool                = "true",
        /// Whether to show experimental rust-analyzer diagnostics that might
        /// have more false positives than usual.
        diagnostics_experimental_enable: bool    = "false",
        /// Map of prefixes to be substituted when parsing diagnostic file paths.
        /// This should be the reverse mapping of what is passed to `rustc` as `--remap-path-prefix`.
        diagnostics_remapPrefix: FxHashMap<String, String> = "{}",
        /// List of warnings that should be displayed with hint severity.
        ///
        /// The warnings will be indicated by faded text or three dots in code
        /// and will not show up in the `Problems Panel`.
        diagnostics_warningsAsHint: Vec<String> = "[]",
        /// List of warnings that should be displayed with info severity.
        ///
        /// The warnings will be indicated by a blue squiggly underline in code
        /// and a blue icon in the `Problems Panel`.
        diagnostics_warningsAsInfo: Vec<String> = "[]",

        /// These directories will be ignored by rust-analyzer. They are
        /// relative to the workspace root, and globs are not supported. You may
        /// also need to add the folders to Code's `files.watcherExclude`.
        files_excludeDirs: Vec<PathBuf> = "[]",
        /// Controls file watching implementation.
        files_watcher: FilesWatcherDef = "\"client\"",

        /// Enables highlighting of related references while the cursor is on `break`, `loop`, `while`, or `for` keywords.
        highlightRelated_breakPoints_enable: bool = "true",
        /// Enables highlighting of all exit points while the cursor is on any `return`, `?`, `fn`, or return type arrow (`->`).
        highlightRelated_exitPoints_enable: bool = "true",
        /// Enables highlighting of related references while the cursor is on any identifier.
        highlightRelated_references_enable: bool = "true",
        /// Enables highlighting of all break points for a loop or block context while the cursor is on any `async` or `await` keywords.
        highlightRelated_yieldPoints_enable: bool = "true",

        /// Whether to show `Debug` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_debug_enable: bool           = "true",
        /// Whether to show HoverActions in Rust files.
        hover_actions_enable: bool          = "true",
        /// Whether to show `Go to Type Definition` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_gotoTypeDef_enable: bool     = "true",
        /// Whether to show `Implementations` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_implementations_enable: bool = "true",
        /// Whether to show `References` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_references_enable: bool      = "false",
        /// Whether to show `Run` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_run_enable: bool             = "true",

        /// Whether to show documentation on hover.
        hover_documentation_enable: bool           = "true",
        /// Whether to show keyword hover popups. Only applies when
        /// `#rust-analyzer.hover.documentation.enable#` is set.
        hover_documentation_keywords_enable: bool  = "true",
        /// Use markdown syntax for links in hover.
        hover_links_enable: bool = "true",

        /// Whether to enforce the import granularity setting for all files. If set to false rust-analyzer will try to keep import styles consistent per file.
        imports_granularity_enforce: bool              = "false",
        /// How imports should be grouped into use statements.
        imports_granularity_group: ImportGranularityDef  = "\"crate\"",
        /// Group inserted imports by the https://rust-analyzer.github.io/manual.html#auto-import[following order]. Groups are separated by newlines.
        imports_group_enable: bool                           = "true",
        /// Whether to allow import insertion to merge new imports into single path glob imports like `use std::fmt::*;`.
        imports_merge_glob: bool           = "true",
        /// The path structure for newly inserted paths to use.
        imports_prefix: ImportPrefixDef               = "\"plain\"",

        /// Whether to show inlay type hints for binding modes.
        inlayHints_bindingModeHints_enable: bool                   = "false",
        /// Whether to show inlay type hints for method chains.
        inlayHints_chainingHints_enable: bool                      = "true",
        /// Whether to show inlay hints after a closing `}` to indicate what item it belongs to.
        inlayHints_closingBraceHints_enable: bool                  = "true",
        /// Minimum number of lines required before the `}` until the hint is shown (set to 0 or 1
        /// to always show them).
        inlayHints_closingBraceHints_minLines: usize               = "25",
        /// Whether to show inlay type hints for return types of closures.
        inlayHints_closureReturnTypeHints_enable: ClosureReturnTypeHintsDef  = "\"never\"",
        /// Whether to show inlay type hints for elided lifetimes in function signatures.
        inlayHints_lifetimeElisionHints_enable: LifetimeElisionDef = "\"never\"",
        /// Whether to prefer using parameter names as the name for elided lifetime hints if possible.
        inlayHints_lifetimeElisionHints_useParameterNames: bool    = "false",
        /// Maximum length for inlay hints. Set to null to have an unlimited length.
        inlayHints_maxLength: Option<usize>                        = "25",
        /// Whether to show function parameter name inlay hints at the call
        /// site.
        inlayHints_parameterHints_enable: bool                     = "true",
        /// Whether to show inlay type hints for compiler inserted reborrows.
        inlayHints_reborrowHints_enable: ReborrowHintsDef          = "\"never\"",
        /// Whether to render leading colons for type hints, and trailing colons for parameter hints.
        inlayHints_renderColons: bool                              = "true",
        /// Whether to show inlay type hints for variables.
        inlayHints_typeHints_enable: bool                          = "true",
        /// Whether to hide inlay type hints for `let` statements that initialize to a closure.
        /// Only applies to closures with blocks, same as `#rust-analyzer.inlayHints.closureReturnTypeHints.enable#`.
        inlayHints_typeHints_hideClosureInitialization: bool       = "false",
        /// Whether to hide inlay type hints for constructors.
        inlayHints_typeHints_hideNamedConstructor: bool            = "false",

        /// Join lines merges consecutive declaration and initialization of an assignment.
        joinLines_joinAssignments: bool = "true",
        /// Join lines inserts else between consecutive ifs.
        joinLines_joinElseIf: bool = "true",
        /// Join lines removes trailing commas.
        joinLines_removeTrailingComma: bool = "true",
        /// Join lines unwraps trivial blocks.
        joinLines_unwrapTrivialBlock: bool = "true",

        /// Whether to show `Debug` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_debug_enable: bool            = "true",
        /// Whether to show CodeLens in Rust files.
        lens_enable: bool           = "true",
        /// Internal config: use custom client-side commands even when the
        /// client doesn't set the corresponding capability.
        lens_forceCustomCommands: bool = "true",
        /// Whether to show `Implementations` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_implementations_enable: bool  = "true",
        /// Whether to show `References` lens for Struct, Enum, and Union.
        /// Only applies when `#rust-analyzer.lens.enable#` is set.
        lens_references_adt_enable: bool = "false",
        /// Whether to show `References` lens for Enum Variants.
        /// Only applies when `#rust-analyzer.lens.enable#` is set.
        lens_references_enumVariant_enable: bool = "false",
        /// Whether to show `Method References` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_references_method_enable: bool = "false",
        /// Whether to show `References` lens for Trait.
        /// Only applies when `#rust-analyzer.lens.enable#` is set.
        lens_references_trait_enable: bool = "false",
        /// Whether to show `Run` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_run_enable: bool              = "true",

        /// Disable project auto-discovery in favor of explicitly specified set
        /// of projects.
        ///
        /// Elements must be paths pointing to `Cargo.toml`,
        /// `rust-project.json`, or JSON objects in `rust-project.json` format.
        linkedProjects: Vec<ManifestOrProjectJson> = "[]",

        /// Number of syntax trees rust-analyzer keeps in memory. Defaults to 128.
        lru_capacity: Option<usize>                 = "null",

        /// Whether to show `can't find Cargo.toml` error message.
        notifications_cargoTomlNotFound: bool      = "true",

        /// Expand attribute macros. Requires `#rust-analyzer.procMacro.enable#` to be set.
        procMacro_attributes_enable: bool = "true",
        /// Enable support for procedural macros, implies `#rust-analyzer.cargo.buildScripts.enable#`.
        procMacro_enable: bool                     = "true",
        /// These proc-macros will be ignored when trying to expand them.
        ///
        /// This config takes a map of crate names with the exported proc-macro names to ignore as values.
        procMacro_ignored: FxHashMap<Box<str>, Box<[Box<str>]>>          = "{}",
        /// Internal config, path to proc-macro server executable (typically,
        /// this is rust-analyzer itself, but we override this in tests).
        procMacro_server: Option<PathBuf>          = "null",

        /// Command to be executed instead of 'cargo' for runnables.
        runnables_command: Option<String> = "null",
        /// Additional arguments to be passed to cargo for runnables such as
        /// tests or binaries. For example, it may be `--release`.
        runnables_extraArgs: Vec<String>   = "[]",

        /// Path to the Cargo.toml of the rust compiler workspace, for usage in rustc_private
        /// projects, or "discover" to try to automatically find it if the `rustc-dev` component
        /// is installed.
        ///
        /// Any project which uses rust-analyzer with the rustcPrivate
        /// crates must set `[package.metadata.rust-analyzer] rustc_private=true` to use it.
        ///
        /// This option does not take effect until rust-analyzer is restarted.
        rustc_source: Option<String> = "null",

        /// Additional arguments to `rustfmt`.
        rustfmt_extraArgs: Vec<String>               = "[]",
        /// Advanced option, fully override the command rust-analyzer uses for
        /// formatting.
        rustfmt_overrideCommand: Option<Vec<String>> = "null",
        /// Enables the use of rustfmt's unstable range formatting command for the
        /// `textDocument/rangeFormatting` request. The rustfmt option is unstable and only
        /// available on a nightly build.
        rustfmt_rangeFormatting_enable: bool = "false",

        /// Use semantic tokens for strings.
        ///
        /// In some editors (e.g. vscode) semantic tokens override other highlighting grammars.
        /// By disabling semantic tokens for strings, other grammars can be used to highlight
        /// their contents.
        semanticHighlighting_strings_enable: bool = "true",

        /// Show full signature of the callable. Only shows parameters if disabled.
        signatureInfo_detail: SignatureDetail                           = "\"full\"",
        /// Show documentation.
        signatureInfo_documentation_enable: bool                       = "true",

        /// Whether to insert closing angle brackets when typing an opening angle bracket of a generic argument list.
        typing_autoClosingAngleBrackets_enable: bool = "false",

        /// Workspace symbol search kind.
        workspace_symbol_search_kind: WorkspaceSymbolSearchKindDef = "\"only_types\"",
        /// Limits the number of items returned from a workspace symbol search (Defaults to 128).
        /// Some clients like vs-code issue new searches on result filtering and don't require all results to be returned in the initial search.
        /// Other clients requires all results upfront and might require a higher limit.
        workspace_symbol_search_limit: usize = "128",
        /// Workspace symbol search scope.
        workspace_symbol_search_scope: WorkspaceSymbolSearchScopeDef = "\"workspace\"",
    }
}

impl Default for ConfigData {
    fn default() -> Self {
        ConfigData::from_json(serde_json::Value::Null, &mut Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub discovered_projects: Option<Vec<ProjectManifest>>,
    caps: lsp_types::ClientCapabilities,
    root_path: AbsPathBuf,
    data: ConfigData,
    detached_files: Vec<AbsPathBuf>,
    snippets: Vec<Snippet>,
}

type ParallelCachePrimingNumThreads = u8;

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

pub struct CallInfoConfig {
    pub params_only: bool,
    pub docs: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LensConfig {
    // runnables
    pub run: bool,
    pub debug: bool,

    // implementations
    pub implementations: bool,

    // references
    pub method_refs: bool,
    pub refs_adt: bool,   // for Struct, Enum, Union and Trait
    pub refs_trait: bool, // for Struct, Enum, Union and Trait
    pub enum_variant_refs: bool,
}

impl LensConfig {
    pub fn any(&self) -> bool {
        self.run
            || self.debug
            || self.implementations
            || self.method_refs
            || self.refs_adt
            || self.refs_trait
            || self.enum_variant_refs
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn runnable(&self) -> bool {
        self.run || self.debug
    }

    pub fn references(&self) -> bool {
        self.method_refs || self.refs_adt || self.refs_trait || self.enum_variant_refs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoverActionsConfig {
    pub implementations: bool,
    pub references: bool,
    pub run: bool,
    pub debug: bool,
    pub goto_type_def: bool,
}

impl HoverActionsConfig {
    pub const NO_ACTIONS: Self = Self {
        implementations: false,
        references: false,
        run: false,
        debug: false,
        goto_type_def: false,
    };

    pub fn any(&self) -> bool {
        self.implementations || self.references || self.runnable() || self.goto_type_def
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
    pub exclude: Vec<AbsPathBuf>,
}

#[derive(Debug, Clone)]
pub enum FilesWatcher {
    Client,
    Server,
}

#[derive(Debug, Clone)]
pub struct NotificationsConfig {
    pub cargo_toml_not_found: bool,
}

#[derive(Debug, Clone)]
pub enum RustfmtConfig {
    Rustfmt { extra_args: Vec<String>, enable_range_formatting: bool },
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

/// Configuration for workspace symbol search requests.
#[derive(Debug, Clone)]
pub struct WorkspaceSymbolConfig {
    /// In what scope should the symbol be searched in.
    pub search_scope: WorkspaceSymbolSearchScope,
    /// What kind of symbol is being searched for.
    pub search_kind: WorkspaceSymbolSearchKind,
    /// How many items are returned at most.
    pub search_limit: usize,
}

pub struct ClientCommandsConfig {
    pub run_single: bool,
    pub debug_single: bool,
    pub show_reference: bool,
    pub goto_location: bool,
    pub trigger_parameter_hints: bool,
}

#[derive(Debug)]
pub struct ConfigUpdateError {
    errors: Vec<(String, serde_json::Error)>,
}

impl fmt::Display for ConfigUpdateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let errors = self.errors.iter().format_with("\n", |(key, e), f| {
            f(key)?;
            f(&": ")?;
            f(e)
        });
        write!(
            f,
            "rust-analyzer found {} invalid config value{}:\n{}",
            self.errors.len(),
            if self.errors.len() == 1 { "" } else { "s" },
            errors
        )
    }
}

impl Config {
    pub fn new(root_path: AbsPathBuf, caps: ClientCapabilities) -> Self {
        Config {
            caps,
            data: ConfigData::default(),
            detached_files: Vec::new(),
            discovered_projects: None,
            root_path,
            snippets: Default::default(),
        }
    }

    pub fn update(&mut self, mut json: serde_json::Value) -> Result<(), ConfigUpdateError> {
        tracing::info!("updating config from JSON: {:#}", json);
        if json.is_null() || json.as_object().map_or(false, |it| it.is_empty()) {
            return Ok(());
        }
        let mut errors = Vec::new();
        self.detached_files =
            get_field::<Vec<PathBuf>>(&mut json, &mut errors, "detachedFiles", None, "[]")
                .into_iter()
                .map(AbsPathBuf::assert)
                .collect();
        patch_old_style::patch_json_for_outdated_configs(&mut json);
        self.data = ConfigData::from_json(json, &mut errors);
        tracing::debug!("deserialized config data: {:#?}", self.data);
        self.snippets.clear();
        for (name, def) in self.data.completion_snippets_custom.iter() {
            if def.prefix.is_empty() && def.postfix.is_empty() {
                continue;
            }
            let scope = match def.scope {
                SnippetScopeDef::Expr => SnippetScope::Expr,
                SnippetScopeDef::Type => SnippetScope::Type,
                SnippetScopeDef::Item => SnippetScope::Item,
            };
            match Snippet::new(
                &def.prefix,
                &def.postfix,
                &def.body,
                def.description.as_ref().unwrap_or(name),
                &def.requires,
                scope,
            ) {
                Some(snippet) => self.snippets.push(snippet),
                None => errors.push((
                    format!("snippet {name} is invalid"),
                    <serde_json::Error as serde::de::Error>::custom(
                        "snippet path is invalid or triggers are missing",
                    ),
                )),
            }
        }

        self.validate(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigUpdateError { errors })
        }
    }

    fn validate(&self, error_sink: &mut Vec<(String, serde_json::Error)>) {
        use serde::de::Error;
        if self.data.checkOnSave_command.is_empty() {
            error_sink.push((
                "/checkOnSave/command".to_string(),
                serde_json::Error::custom("expected a non-empty string"),
            ));
        }
    }

    pub fn json_schema() -> serde_json::Value {
        ConfigData::json_schema()
    }

    pub fn root_path(&self) -> &AbsPathBuf {
        &self.root_path
    }

    pub fn caps(&self) -> &lsp_types::ClientCapabilities {
        &self.caps
    }

    pub fn detached_files(&self) -> &[AbsPathBuf] {
        &self.detached_files
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

macro_rules! try_or_def {
    ($expr:expr) => {
        try_!($expr).unwrap_or_default()
    };
}

impl Config {
    pub fn linked_projects(&self) -> Vec<LinkedProject> {
        match self.data.linkedProjects.as_slice() {
            [] => match self.discovered_projects.as_ref() {
                Some(discovered_projects) => {
                    let exclude_dirs: Vec<_> = self
                        .data
                        .files_excludeDirs
                        .iter()
                        .map(|p| self.root_path.join(p))
                        .collect();
                    discovered_projects
                        .iter()
                        .filter(|p| {
                            let (ProjectManifest::ProjectJson(path)
                            | ProjectManifest::CargoToml(path)) = p;
                            !exclude_dirs.iter().any(|p| path.starts_with(p))
                        })
                        .cloned()
                        .map(LinkedProject::from)
                        .collect()
                }
                None => Vec::new(),
            },
            linked_projects => linked_projects
                .iter()
                .filter_map(|linked_project| match linked_project {
                    ManifestOrProjectJson::Manifest(it) => {
                        let path = self.root_path.join(it);
                        ProjectManifest::from_manifest_file(path)
                            .map_err(|e| tracing::error!("failed to load linked project: {}", e))
                            .ok()
                            .map(Into::into)
                    }
                    ManifestOrProjectJson::ProjectJson(it) => {
                        Some(ProjectJson::new(&self.root_path, it.clone()).into())
                    }
                })
                .collect(),
        }
    }

    pub fn did_save_text_document_dynamic_registration(&self) -> bool {
        let caps = try_or_def!(self.caps.text_document.as_ref()?.synchronization.clone()?);
        caps.did_save == Some(true) && caps.dynamic_registration == Some(true)
    }

    pub fn did_change_watched_files_dynamic_registration(&self) -> bool {
        try_or_def!(
            self.caps.workspace.as_ref()?.did_change_watched_files.as_ref()?.dynamic_registration?
        )
    }

    pub fn prefill_caches(&self) -> bool {
        self.data.cachePriming_enable
    }

    pub fn location_link(&self) -> bool {
        try_or_def!(self.caps.text_document.as_ref()?.definition?.link_support?)
    }

    pub fn line_folding_only(&self) -> bool {
        try_or_def!(self.caps.text_document.as_ref()?.folding_range.as_ref()?.line_folding_only?)
    }

    pub fn hierarchical_symbols(&self) -> bool {
        try_or_def!(
            self.caps
                .text_document
                .as_ref()?
                .document_symbol
                .as_ref()?
                .hierarchical_document_symbol_support?
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
        try_or_def!(self.caps.window.as_ref()?.work_done_progress?)
    }

    pub fn will_rename(&self) -> bool {
        try_or_def!(self.caps.workspace.as_ref()?.file_operations.as_ref()?.will_rename?)
    }

    pub fn change_annotation_support(&self) -> bool {
        try_!(self
            .caps
            .workspace
            .as_ref()?
            .workspace_edit
            .as_ref()?
            .change_annotation_support
            .as_ref()?)
        .is_some()
    }

    pub fn code_action_resolve(&self) -> bool {
        try_or_def!(self
            .caps
            .text_document
            .as_ref()?
            .code_action
            .as_ref()?
            .resolve_support
            .as_ref()?
            .properties
            .as_slice())
        .iter()
        .any(|it| it == "edit")
    }

    pub fn signature_help_label_offsets(&self) -> bool {
        try_or_def!(
            self.caps
                .text_document
                .as_ref()?
                .signature_help
                .as_ref()?
                .signature_information
                .as_ref()?
                .parameter_information
                .as_ref()?
                .label_offset_support?
        )
    }

    pub fn completion_label_details_support(&self) -> bool {
        try_!(self
            .caps
            .text_document
            .as_ref()?
            .completion
            .as_ref()?
            .completion_item
            .as_ref()?
            .label_details_support
            .as_ref()?)
        .is_some()
    }

    pub fn offset_encoding(&self) -> OffsetEncoding {
        if supports_utf8(&self.caps) {
            OffsetEncoding::Utf8
        } else {
            OffsetEncoding::Utf16
        }
    }

    fn experimental(&self, index: &'static str) -> bool {
        try_or_def!(self.caps.experimental.as_ref()?.get(index)?.as_bool()?)
    }

    pub fn code_action_group(&self) -> bool {
        self.experimental("codeActionGroup")
    }

    pub fn server_status_notification(&self) -> bool {
        self.experimental("serverStatusNotification")
    }

    pub fn publish_diagnostics(&self) -> bool {
        self.data.diagnostics_enable
    }

    pub fn diagnostics(&self) -> DiagnosticsConfig {
        DiagnosticsConfig {
            proc_attr_macros_enabled: self.expand_proc_attr_macros(),
            proc_macros_enabled: self.data.procMacro_enable,
            disable_experimental: !self.data.diagnostics_experimental_enable,
            disabled: self.data.diagnostics_disabled.clone(),
            expr_fill_default: match self.data.assist_expressionFillDefault {
                ExprFillDefaultDef::Todo => ExprFillDefaultMode::Todo,
                ExprFillDefaultDef::Default => ExprFillDefaultMode::Default,
            },
            insert_use: self.insert_use_config(),
        }
    }

    pub fn diagnostics_map(&self) -> DiagnosticsMapConfig {
        DiagnosticsMapConfig {
            remap_prefix: self.data.diagnostics_remapPrefix.clone(),
            warnings_as_info: self.data.diagnostics_warningsAsInfo.clone(),
            warnings_as_hint: self.data.diagnostics_warningsAsHint.clone(),
        }
    }

    pub fn lru_capacity(&self) -> Option<usize> {
        self.data.lru_capacity
    }

    pub fn proc_macro_srv(&self) -> Option<(AbsPathBuf, Vec<OsString>)> {
        if !self.data.procMacro_enable {
            return None;
        }
        let path = match &self.data.procMacro_server {
            Some(it) => self.root_path.join(it),
            None => AbsPathBuf::assert(std::env::current_exe().ok()?),
        };
        Some((path, vec!["proc-macro".into()]))
    }

    pub fn dummy_replacements(&self) -> &FxHashMap<Box<str>, Box<[Box<str>]>> {
        &self.data.procMacro_ignored
    }

    pub fn expand_proc_attr_macros(&self) -> bool {
        self.data.procMacro_enable && self.data.procMacro_attributes_enable
    }

    pub fn files(&self) -> FilesConfig {
        FilesConfig {
            watcher: match self.data.files_watcher {
                FilesWatcherDef::Client if self.did_change_watched_files_dynamic_registration() => {
                    FilesWatcher::Client
                }
                _ => FilesWatcher::Server,
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
        self.data.cargo_buildScripts_enable || self.data.procMacro_enable
    }

    pub fn cargo(&self) -> CargoConfig {
        let rustc_source = self.data.rustc_source.as_ref().map(|rustc_src| {
            if rustc_src == "discover" {
                RustcSource::Discover
            } else {
                RustcSource::Path(self.root_path.join(rustc_src))
            }
        });

        CargoConfig {
            no_default_features: self.data.cargo_noDefaultFeatures,
            all_features: matches!(self.data.cargo_features, CargoFeatures::All),
            features: match &self.data.cargo_features {
                CargoFeatures::All => vec![],
                CargoFeatures::Listed(it) => it.clone(),
            },
            target: self.data.cargo_target.clone(),
            no_sysroot: self.data.cargo_noSysroot,
            rustc_source,
            unset_test_crates: UnsetTestCrates::Only(self.data.cargo_unsetTest.clone()),
            wrap_rustc_in_build_scripts: self.data.cargo_buildScripts_useRustcWrapper,
            run_build_script_command: self.data.cargo_buildScripts_overrideCommand.clone(),
        }
    }

    pub fn rustfmt(&self) -> RustfmtConfig {
        match &self.data.rustfmt_overrideCommand {
            Some(args) if !args.is_empty() => {
                let mut args = args.clone();
                let command = args.remove(0);
                RustfmtConfig::CustomCommand { command, args }
            }
            Some(_) | None => RustfmtConfig::Rustfmt {
                extra_args: self.data.rustfmt_extraArgs.clone(),
                enable_range_formatting: self.data.rustfmt_rangeFormatting_enable,
            },
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
                all_features: matches!(
                    self.data.checkOnSave_features.as_ref().unwrap_or(&self.data.cargo_features),
                    CargoFeatures::All
                ),
                features: match self
                    .data
                    .checkOnSave_features
                    .clone()
                    .unwrap_or_else(|| self.data.cargo_features.clone())
                {
                    CargoFeatures::All => vec![],
                    CargoFeatures::Listed(it) => it,
                },
                extra_args: self.data.checkOnSave_extraArgs.clone(),
            },
        };
        Some(flycheck_config)
    }

    pub fn runnables(&self) -> RunnablesConfig {
        RunnablesConfig {
            override_cargo: self.data.runnables_command.clone(),
            cargo_extra_args: self.data.runnables_extraArgs.clone(),
        }
    }

    pub fn inlay_hints(&self) -> InlayHintsConfig {
        InlayHintsConfig {
            render_colons: self.data.inlayHints_renderColons,
            type_hints: self.data.inlayHints_typeHints_enable,
            parameter_hints: self.data.inlayHints_parameterHints_enable,
            chaining_hints: self.data.inlayHints_chainingHints_enable,
            closure_return_type_hints: match self.data.inlayHints_closureReturnTypeHints_enable {
                ClosureReturnTypeHintsDef::Always => ide::ClosureReturnTypeHints::Always,
                ClosureReturnTypeHintsDef::Never => ide::ClosureReturnTypeHints::Never,
                ClosureReturnTypeHintsDef::WithBlock => ide::ClosureReturnTypeHints::WithBlock,
            },
            lifetime_elision_hints: match self.data.inlayHints_lifetimeElisionHints_enable {
                LifetimeElisionDef::Always => ide::LifetimeElisionHints::Always,
                LifetimeElisionDef::Never => ide::LifetimeElisionHints::Never,
                LifetimeElisionDef::SkipTrivial => ide::LifetimeElisionHints::SkipTrivial,
            },
            hide_named_constructor_hints: self.data.inlayHints_typeHints_hideNamedConstructor,
            hide_closure_initialization_hints: self
                .data
                .inlayHints_typeHints_hideClosureInitialization,
            reborrow_hints: match self.data.inlayHints_reborrowHints_enable {
                ReborrowHintsDef::Always => ide::ReborrowHints::Always,
                ReborrowHintsDef::Never => ide::ReborrowHints::Never,
                ReborrowHintsDef::Mutable => ide::ReborrowHints::MutableOnly,
            },
            binding_mode_hints: self.data.inlayHints_bindingModeHints_enable,
            param_names_for_lifetime_elision_hints: self
                .data
                .inlayHints_lifetimeElisionHints_useParameterNames,
            max_length: self.data.inlayHints_maxLength,
            closing_brace_hints_min_lines: if self.data.inlayHints_closingBraceHints_enable {
                Some(self.data.inlayHints_closingBraceHints_minLines)
            } else {
                None
            },
        }
    }

    fn insert_use_config(&self) -> InsertUseConfig {
        InsertUseConfig {
            granularity: match self.data.imports_granularity_group {
                ImportGranularityDef::Preserve => ImportGranularity::Preserve,
                ImportGranularityDef::Item => ImportGranularity::Item,
                ImportGranularityDef::Crate => ImportGranularity::Crate,
                ImportGranularityDef::Module => ImportGranularity::Module,
            },
            enforce_granularity: self.data.imports_granularity_enforce,
            prefix_kind: match self.data.imports_prefix {
                ImportPrefixDef::Plain => PrefixKind::Plain,
                ImportPrefixDef::ByCrate => PrefixKind::ByCrate,
                ImportPrefixDef::BySelf => PrefixKind::BySelf,
            },
            group: self.data.imports_group_enable,
            skip_glob_imports: !self.data.imports_merge_glob,
        }
    }

    pub fn completion(&self) -> CompletionConfig {
        CompletionConfig {
            enable_postfix_completions: self.data.completion_postfix_enable,
            enable_imports_on_the_fly: self.data.completion_autoimport_enable
                && completion_item_edit_resolve(&self.caps),
            enable_self_on_the_fly: self.data.completion_autoself_enable,
            enable_private_editable: self.data.completion_privateEditable_enable,
            callable: match self.data.completion_callable_snippets {
                CallableCompletionDef::FillArguments => Some(CallableSnippets::FillArguments),
                CallableCompletionDef::AddParentheses => Some(CallableSnippets::AddParentheses),
                CallableCompletionDef::None => None,
            },
            insert_use: self.insert_use_config(),
            snippet_cap: SnippetCap::new(try_or_def!(
                self.caps
                    .text_document
                    .as_ref()?
                    .completion
                    .as_ref()?
                    .completion_item
                    .as_ref()?
                    .snippet_support?
            )),
            snippets: self.snippets.clone(),
        }
    }

    pub fn snippet_cap(&self) -> bool {
        self.experimental("snippetTextEdit")
    }

    pub fn assist(&self) -> AssistConfig {
        AssistConfig {
            snippet_cap: SnippetCap::new(self.experimental("snippetTextEdit")),
            allowed: None,
            insert_use: self.insert_use_config(),
        }
    }

    pub fn join_lines(&self) -> JoinLinesConfig {
        JoinLinesConfig {
            join_else_if: self.data.joinLines_joinElseIf,
            remove_trailing_comma: self.data.joinLines_removeTrailingComma,
            unwrap_trivial_blocks: self.data.joinLines_unwrapTrivialBlock,
            join_assignments: self.data.joinLines_joinAssignments,
        }
    }

    pub fn call_info(&self) -> CallInfoConfig {
        CallInfoConfig {
            params_only: matches!(self.data.signatureInfo_detail, SignatureDetail::Parameters),
            docs: self.data.signatureInfo_documentation_enable,
        }
    }

    pub fn lens(&self) -> LensConfig {
        LensConfig {
            run: self.data.lens_enable && self.data.lens_run_enable,
            debug: self.data.lens_enable && self.data.lens_debug_enable,
            implementations: self.data.lens_enable && self.data.lens_implementations_enable,
            method_refs: self.data.lens_enable && self.data.lens_references_method_enable,
            refs_adt: self.data.lens_enable && self.data.lens_references_adt_enable,
            refs_trait: self.data.lens_enable && self.data.lens_references_trait_enable,
            enum_variant_refs: self.data.lens_enable
                && self.data.lens_references_enumVariant_enable,
        }
    }

    pub fn hover_actions(&self) -> HoverActionsConfig {
        let enable = self.experimental("hoverActions") && self.data.hover_actions_enable;
        HoverActionsConfig {
            implementations: enable && self.data.hover_actions_implementations_enable,
            references: enable && self.data.hover_actions_references_enable,
            run: enable && self.data.hover_actions_run_enable,
            debug: enable && self.data.hover_actions_debug_enable,
            goto_type_def: enable && self.data.hover_actions_gotoTypeDef_enable,
        }
    }

    pub fn highlighting_strings(&self) -> bool {
        self.data.semanticHighlighting_strings_enable
    }

    pub fn hover(&self) -> HoverConfig {
        HoverConfig {
            links_in_hover: self.data.hover_links_enable,
            documentation: self.data.hover_documentation_enable.then(|| {
                let is_markdown = try_or_def!(self
                    .caps
                    .text_document
                    .as_ref()?
                    .hover
                    .as_ref()?
                    .content_format
                    .as_ref()?
                    .as_slice())
                .contains(&MarkupKind::Markdown);
                if is_markdown {
                    HoverDocFormat::Markdown
                } else {
                    HoverDocFormat::PlainText
                }
            }),
            keywords: self.data.hover_documentation_keywords_enable,
        }
    }

    pub fn workspace_symbol(&self) -> WorkspaceSymbolConfig {
        WorkspaceSymbolConfig {
            search_scope: match self.data.workspace_symbol_search_scope {
                WorkspaceSymbolSearchScopeDef::Workspace => WorkspaceSymbolSearchScope::Workspace,
                WorkspaceSymbolSearchScopeDef::WorkspaceAndDependencies => {
                    WorkspaceSymbolSearchScope::WorkspaceAndDependencies
                }
            },
            search_kind: match self.data.workspace_symbol_search_kind {
                WorkspaceSymbolSearchKindDef::OnlyTypes => WorkspaceSymbolSearchKind::OnlyTypes,
                WorkspaceSymbolSearchKindDef::AllSymbols => WorkspaceSymbolSearchKind::AllSymbols,
            },
            search_limit: self.data.workspace_symbol_search_limit,
        }
    }

    pub fn semantic_tokens_refresh(&self) -> bool {
        try_or_def!(self.caps.workspace.as_ref()?.semantic_tokens.as_ref()?.refresh_support?)
    }

    pub fn code_lens_refresh(&self) -> bool {
        try_or_def!(self.caps.workspace.as_ref()?.code_lens.as_ref()?.refresh_support?)
    }

    pub fn insert_replace_support(&self) -> bool {
        try_or_def!(
            self.caps
                .text_document
                .as_ref()?
                .completion
                .as_ref()?
                .completion_item
                .as_ref()?
                .insert_replace_support?
        )
    }

    pub fn client_commands(&self) -> ClientCommandsConfig {
        let commands =
            try_or!(self.caps.experimental.as_ref()?.get("commands")?, &serde_json::Value::Null);
        let commands: Option<lsp_ext::ClientCommandOptions> =
            serde_json::from_value(commands.clone()).ok();
        let force = commands.is_none() && self.data.lens_forceCustomCommands;
        let commands = commands.map(|it| it.commands).unwrap_or_default();

        let get = |name: &str| commands.iter().any(|it| it == name) || force;

        ClientCommandsConfig {
            run_single: get("rust-analyzer.runSingle"),
            debug_single: get("rust-analyzer.debugSingle"),
            show_reference: get("rust-analyzer.showReferences"),
            goto_location: get("rust-analyzer.gotoLocation"),
            trigger_parameter_hints: get("editor.action.triggerParameterHints"),
        }
    }

    pub fn highlight_related(&self) -> HighlightRelatedConfig {
        HighlightRelatedConfig {
            references: self.data.highlightRelated_references_enable,
            break_points: self.data.highlightRelated_breakPoints_enable,
            exit_points: self.data.highlightRelated_exitPoints_enable,
            yield_points: self.data.highlightRelated_yieldPoints_enable,
        }
    }

    pub fn prime_caches_num_threads(&self) -> u8 {
        match self.data.cachePriming_numThreads {
            0 => num_cpus::get_physical().try_into().unwrap_or(u8::MAX),
            n => n,
        }
    }

    pub fn typing_autoclose_angle(&self) -> bool {
        self.data.typing_autoClosingAngleBrackets_enable
    }
}
// Deserialization definitions

macro_rules! create_bool_or_string_de {
    ($ident:ident<$bool:literal, $string:literal>) => {
        fn $ident<'de, D>(d: D) -> Result<(), D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct V;
            impl<'de> serde::de::Visitor<'de> for V {
                type Value = ();

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str(concat!(
                        stringify!($bool),
                        " or \"",
                        stringify!($string),
                        "\""
                    ))
                }

                fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    match v {
                        $bool => Ok(()),
                        _ => Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Bool(v),
                            &self,
                        )),
                    }
                }

                fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    match v {
                        $string => Ok(()),
                        _ => Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(v),
                            &self,
                        )),
                    }
                }

                fn visit_enum<A>(self, a: A) -> Result<Self::Value, A::Error>
                where
                    A: serde::de::EnumAccess<'de>,
                {
                    use serde::de::VariantAccess;
                    let (variant, va) = a.variant::<&'de str>()?;
                    va.unit_variant()?;
                    match variant {
                        $string => Ok(()),
                        _ => Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(variant),
                            &self,
                        )),
                    }
                }
            }
            d.deserialize_any(V)
        }
    };
}
create_bool_or_string_de!(true_or_always<true, "always">);
create_bool_or_string_de!(false_or_never<false, "never">);

macro_rules! named_unit_variant {
    ($variant:ident) => {
        pub(super) fn $variant<'de, D>(deserializer: D) -> Result<(), D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct V;
            impl<'de> serde::de::Visitor<'de> for V {
                type Value = ();
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str(concat!("\"", stringify!($variant), "\""))
                }
                fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                    if value == stringify!($variant) {
                        Ok(())
                    } else {
                        Err(E::invalid_value(serde::de::Unexpected::Str(value), &self))
                    }
                }
            }
            deserializer.deserialize_str(V)
        }
    };
}

mod de_unit_v {
    named_unit_variant!(all);
    named_unit_variant!(skip_trivial);
    named_unit_variant!(mutable);
    named_unit_variant!(with_block);
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum SnippetScopeDef {
    Expr,
    Item,
    Type,
}

impl Default for SnippetScopeDef {
    fn default() -> Self {
        SnippetScopeDef::Expr
    }
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
struct SnippetDef {
    #[serde(deserialize_with = "single_or_array")]
    prefix: Vec<String>,
    #[serde(deserialize_with = "single_or_array")]
    postfix: Vec<String>,
    description: Option<String>,
    #[serde(deserialize_with = "single_or_array")]
    body: Vec<String>,
    #[serde(deserialize_with = "single_or_array")]
    requires: Vec<String>,
    scope: SnippetScopeDef,
}

fn single_or_array<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct SingleOrVec;

    impl<'de> serde::de::Visitor<'de> for SingleOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("string or array of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            Deserialize::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(SingleOrVec)
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum ManifestOrProjectJson {
    Manifest(PathBuf),
    ProjectJson(ProjectJsonData),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ExprFillDefaultDef {
    Todo,
    Default,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ImportGranularityDef {
    Preserve,
    Item,
    Crate,
    Module,
}

#[derive(Deserialize, Debug, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum CallableCompletionDef {
    FillArguments,
    AddParentheses,
    None,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum CargoFeatures {
    #[serde(deserialize_with = "de_unit_v::all")]
    All,
    Listed(Vec<String>),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum LifetimeElisionDef {
    #[serde(deserialize_with = "true_or_always")]
    Always,
    #[serde(deserialize_with = "false_or_never")]
    Never,
    #[serde(deserialize_with = "de_unit_v::skip_trivial")]
    SkipTrivial,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum ClosureReturnTypeHintsDef {
    #[serde(deserialize_with = "true_or_always")]
    Always,
    #[serde(deserialize_with = "false_or_never")]
    Never,
    #[serde(deserialize_with = "de_unit_v::with_block")]
    WithBlock,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum ReborrowHintsDef {
    #[serde(deserialize_with = "true_or_always")]
    Always,
    #[serde(deserialize_with = "false_or_never")]
    Never,
    #[serde(deserialize_with = "de_unit_v::mutable")]
    Mutable,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum FilesWatcherDef {
    Client,
    Notify,
    Server,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ImportPrefixDef {
    Plain,
    #[serde(alias = "self")]
    BySelf,
    #[serde(alias = "crate")]
    ByCrate,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum WorkspaceSymbolSearchScopeDef {
    Workspace,
    WorkspaceAndDependencies,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum SignatureDetail {
    Full,
    Parameters,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum WorkspaceSymbolSearchKindDef {
    OnlyTypes,
    AllSymbols,
}

macro_rules! _config_data {
    (struct $name:ident {
        $(
            $(#[doc=$doc:literal])*
            $field:ident $(| $alias:ident)*: $ty:ty = $default:expr,
        )*
    }) => {
        #[allow(non_snake_case)]
        #[derive(Debug, Clone)]
        struct $name { $($field: $ty,)* }
        impl $name {
            fn from_json(mut json: serde_json::Value, error_sink: &mut Vec<(String, serde_json::Error)>) -> $name {
                $name {$(
                    $field: get_field(
                        &mut json,
                        error_sink,
                        stringify!($field),
                        None$(.or(Some(stringify!($alias))))*,
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

        #[test]
        fn fields_are_sorted() {
            [$(stringify!($field)),*].windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {} does not hold", w[0], w[1]));
        }
    };
}
use _config_data as config_data;

fn get_field<T: DeserializeOwned>(
    json: &mut serde_json::Value,
    error_sink: &mut Vec<(String, serde_json::Error)>,
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
            json.pointer_mut(&pointer).and_then(|it| match serde_json::from_value(it.take()) {
                Ok(it) => Some(it),
                Err(e) => {
                    tracing::warn!("Failed to deserialize config field at {}: {:?}", pointer, e);
                    error_sink.push((pointer, e));
                    None
                }
            })
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
            let name = field.replace('_', ".");
            let name = format!("rust-analyzer.{}", name);
            let props = field_props(field, ty, doc, default);
            (name, props)
        })
        .collect::<serde_json::Map<_, _>>();
    map.into()
}

fn field_props(field: &str, ty: &str, doc: &[&str], default: &str) -> serde_json::Value {
    let doc = doc_comment_to_string(doc);
    let doc = doc.trim_end_matches('\n');
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
        "usize" => set!("type": "integer", "minimum": 0),
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
        "FxHashMap<Box<str>, Box<[Box<str>]>>" => set! {
            "type": "object",
        },
        "FxHashMap<String, SnippetDef>" => set! {
            "type": "object",
        },
        "FxHashMap<String, String>" => set! {
            "type": "object",
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
            "enum": ["none", "crate", "module"],
            "enumDescriptions": [
                "Do not merge imports at all.",
                "Merge imports from the same crate into a single `use` statement.",
                "Merge imports from the same module into a single `use` statement."
            ],
        },
        "ExprFillDefaultDef" => set! {
            "type": "string",
            "enum": ["todo", "default"],
            "enumDescriptions": [
                "Fill missing expressions with the `todo` macro",
                "Fill missing expressions with reasonable defaults, `new` or `default` constructors."
            ],
        },
        "ImportGranularityDef" => set! {
            "type": "string",
            "enum": ["preserve", "crate", "module", "item"],
            "enumDescriptions": [
                "Do not change the granularity of any imports and preserve the original structure written by the developer.",
                "Merge imports from the same crate into a single use statement. Conversely, imports from different crates are split into separate statements.",
                "Merge imports from the same module into a single use statement. Conversely, imports from different modules are split into separate statements.",
                "Flatten imports so that each has its own use statement."
            ],
        },
        "ImportPrefixDef" => set! {
            "type": "string",
            "enum": [
                "plain",
                "self",
                "crate"
            ],
            "enumDescriptions": [
                "Insert import paths relative to the current module, using up to one `super` prefix if the parent module contains the requested item.",
                "Insert import paths relative to the current module, using up to one `super` prefix if the parent module contains the requested item. Prefixes `self` in front of the path if it starts with a module.",
                "Force import paths to be absolute by always starting them with `crate` or the extern crate name they come from."
            ],
        },
        "Vec<ManifestOrProjectJson>" => set! {
            "type": "array",
            "items": { "type": ["string", "object"] },
        },
        "WorkspaceSymbolSearchScopeDef" => set! {
            "type": "string",
            "enum": ["workspace", "workspace_and_dependencies"],
            "enumDescriptions": [
                "Search in current workspace only.",
                "Search in current workspace and dependencies."
            ],
        },
        "WorkspaceSymbolSearchKindDef" => set! {
            "type": "string",
            "enum": ["only_types", "all_symbols"],
            "enumDescriptions": [
                "Search for types only.",
                "Search for all symbols kinds."
            ],
        },
        "ParallelCachePrimingNumThreads" => set! {
            "type": "number",
            "minimum": 0,
            "maximum": 255
        },
        "LifetimeElisionDef" => set! {
            "type": "string",
            "enum": [
                "always",
                "never",
                "skip_trivial"
            ],
            "enumDescriptions": [
                "Always show lifetime elision hints.",
                "Never show lifetime elision hints.",
                "Only show lifetime elision hints if a return type is involved."
            ]
        },
        "ClosureReturnTypeHintsDef" => set! {
            "type": "string",
            "enum": [
                "always",
                "never",
                "with_block"
            ],
            "enumDescriptions": [
                "Always show type hints for return types of closures.",
                "Never show type hints for return types of closures.",
                "Only show type hints for return types of closures with blocks."
            ]
        },
        "ReborrowHintsDef" => set! {
            "type": "string",
            "enum": [
                "always",
                "never",
                "mutable"
            ],
            "enumDescriptions": [
                "Always show reborrow hints.",
                "Never show reborrow hints.",
                "Only show mutable reborrow hints."
            ]
        },
        "CargoFeatures" => set! {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "all"
                    ],
                    "enumDescriptions": [
                        "Pass `--all-features` to cargo",
                    ]
                },
                {
                    "type": "array",
                    "items": { "type": "string" }
                }
            ],
        },
        "Option<CargoFeatures>" => set! {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "all"
                    ],
                    "enumDescriptions": [
                        "Pass `--all-features` to cargo",
                    ]
                },
                {
                    "type": "array",
                    "items": { "type": "string" }
                },
                { "type": "null" }
            ],
        },
        "CallableCompletionDef" => set! {
            "type": "string",
            "enum": [
                "fill_arguments",
                "add_parentheses",
                "none",
            ],
            "enumDescriptions": [
                "Add call parentheses and pre-fill arguments.",
                "Add call parentheses.",
                "Do no snippet completions for callables."
            ]
        },
        "SignatureDetail" => set! {
            "type": "string",
            "enum": ["full", "parameters"],
            "enumDescriptions": [
                "Show the entire signature.",
                "Show only the parameters."
            ],
        },
        "FilesWatcherDef" => set! {
            "type": "string",
            "enum": ["client", "server"],
            "enumDescriptions": [
                "Use the client (editor) to watch files for changes",
                "Use server-side file watching",
            ],
        },
        _ => panic!("missing entry for {}: {}", ty, default),
    }

    map.into()
}

#[cfg(test)]
fn manual(fields: &[(&'static str, &'static str, &[&str], &str)]) -> String {
    fields
        .iter()
        .map(|(field, _ty, doc, default)| {
            let name = format!("rust-analyzer.{}", field.replace('_', "."));
            let doc = doc_comment_to_string(*doc);
            if default.contains('\n') {
                format!(
                    r#"[[{}]]{}::
+
--
Default:
----
{}
----
{}
--
"#,
                    name, name, default, doc
                )
            } else {
                format!("[[{}]]{} (default: `{}`)::\n+\n--\n{}--\n", name, name, default, doc)
            }
        })
        .collect::<String>()
}

fn doc_comment_to_string(doc: &[&str]) -> String {
    doc.iter().map(|it| it.strip_prefix(' ').unwrap_or(it)).map(|it| format!("{}\n", it)).collect()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use test_utils::{ensure_file_contents, project_root};

    use super::*;

    #[test]
    fn generate_package_json_config() {
        let s = Config::json_schema();
        let schema = format!("{:#}", s);
        let mut schema = schema
            .trim_start_matches('{')
            .trim_end_matches('}')
            .replace("  ", "    ")
            .replace('\n', "\n            ")
            .trim_start_matches('\n')
            .trim_end()
            .to_string();
        schema.push_str(",\n");

        // Transform the asciidoc form link to markdown style.
        //
        // https://link[text] => [text](https://link)
        let url_matches = schema.match_indices("https://");
        let mut url_offsets = url_matches.map(|(idx, _)| idx).collect::<Vec<usize>>();
        url_offsets.reverse();
        for idx in url_offsets {
            let link = &schema[idx..];
            // matching on whitespace to ignore normal links
            if let Some(link_end) = link.find(|c| c == ' ' || c == '[') {
                if link.chars().nth(link_end) == Some('[') {
                    if let Some(link_text_end) = link.find(']') {
                        let link_text = link[link_end..(link_text_end + 1)].to_string();

                        schema.replace_range((idx + link_end)..(idx + link_text_end + 1), "");
                        schema.insert(idx, '(');
                        schema.insert(idx + link_end + 1, ')');
                        schema.insert_str(idx, &link_text);
                    }
                }
            }
        }

        let package_json_path = project_root().join("editors/code/package.json");
        let mut package_json = fs::read_to_string(&package_json_path).unwrap();

        let start_marker = "                \"$generated-start\": {},\n";
        let end_marker = "                \"$generated-end\": {}\n";

        let start = package_json.find(start_marker).unwrap() + start_marker.len();
        let end = package_json.find(end_marker).unwrap();

        let p = remove_ws(&package_json[start..end]);
        let s = remove_ws(&schema);
        if !p.contains(&s) {
            package_json.replace_range(start..end, &schema);
            ensure_file_contents(&package_json_path, &package_json)
        }
    }

    #[test]
    fn generate_config_documentation() {
        let docs_path = project_root().join("docs/user/generated_config.adoc");
        let expected = ConfigData::manual();
        ensure_file_contents(&docs_path, &expected);
    }

    fn remove_ws(text: &str) -> String {
        text.replace(char::is_whitespace, "")
    }
}
