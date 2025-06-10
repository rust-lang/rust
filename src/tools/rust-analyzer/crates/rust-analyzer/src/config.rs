//! Config used by the language server.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.
use std::{env, fmt, iter, ops::Not, sync::OnceLock};

use cfg::{CfgAtom, CfgDiff};
use hir::Symbol;
use ide::{
    AssistConfig, CallHierarchyConfig, CallableSnippets, CompletionConfig,
    CompletionFieldsToResolve, DiagnosticsConfig, GenericParameterHints, HighlightConfig,
    HighlightRelatedConfig, HoverConfig, HoverDocFormat, InlayFieldsToResolve, InlayHintsConfig,
    JoinLinesConfig, MemoryLayoutHoverConfig, MemoryLayoutHoverRenderKind, Snippet, SnippetScope,
    SourceRootId,
};
use ide_db::{
    SnippetCap,
    assists::ExprFillDefaultMode,
    imports::insert_use::{ImportGranularity, InsertUseConfig, PrefixKind},
};
use itertools::{Either, Itertools};
use paths::{Utf8Path, Utf8PathBuf};
use project_model::{
    CargoConfig, CargoFeatures, ProjectJson, ProjectJsonData, ProjectJsonFromCommand,
    ProjectManifest, RustLibSource,
};
use rustc_hash::{FxHashMap, FxHashSet};
use semver::Version;
use serde::{
    Deserialize, Serialize,
    de::{DeserializeOwned, Error},
};
use stdx::format_to_acc;
use triomphe::Arc;
use vfs::{AbsPath, AbsPathBuf, VfsPath};

use crate::{
    diagnostics::DiagnosticsMapConfig,
    flycheck::{CargoOptions, FlycheckConfig},
    lsp::capabilities::ClientCapabilities,
    lsp_ext::{WorkspaceSymbolSearchKind, WorkspaceSymbolSearchScope},
};

type FxIndexMap<K, V> = indexmap::IndexMap<K, V, rustc_hash::FxBuildHasher>;

mod patch_old_style;

// Conventions for configuration keys to preserve maximal extendability without breakage:
//  - Toggles (be it binary true/false or with more options in-between) should almost always suffix as `_enable`
//    This has the benefit of namespaces being extensible, and if the suffix doesn't fit later it can be changed without breakage.
//  - In general be wary of using the namespace of something verbatim, it prevents us from adding subkeys in the future
//  - Don't use abbreviations unless really necessary
//  - foo_command = overrides the subcommand, foo_overrideCommand allows full overwriting, extra args only applies for foo_command

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MaxSubstitutionLength {
    Hide,
    #[serde(untagged)]
    Limit(usize),
}

// Defines the server-side configuration of the rust-analyzer. We generate
// *parts* of VS Code's `package.json` config from this. Run `cargo test` to
// re-generate that file.
//
// However, editor specific config, which the server doesn't know about, should
// be specified directly in `package.json`.
//
// To deprecate an option by replacing it with another name use `new_name` | `old_name` so that we keep
// parsing the old name.
config_data! {
    /// Configs that apply on a workspace-wide scope. There are 2 levels on which a global configuration can be configured
    ///
    /// 1. `rust-analyzer.toml` file under user's config directory (e.g ~/.config/rust-analyzer/rust-analyzer.toml)
    /// 2. Client's own configurations (e.g `settings.json` on VS Code)
    ///
    /// A config is searched for by traversing a "config tree" in a bottom up fashion. It is chosen by the nearest first principle.
    global: struct GlobalDefaultConfigData <- GlobalConfigInput -> {
        /// Warm up caches on project load.
        cachePriming_enable: bool = true,
        /// How many worker threads to handle priming caches. The default `0` means to pick automatically.
        cachePriming_numThreads: NumThreads = NumThreads::Physical,

        /// Custom completion snippets.
        completion_snippets_custom: FxIndexMap<String, SnippetDef> = Config::completion_snippets_default(),


        /// These paths (file/directories) will be ignored by rust-analyzer. They are
        /// relative to the workspace root, and globs are not supported. You may
        /// also need to add the folders to Code's `files.watcherExclude`.
        files_exclude | files_excludeDirs: Vec<Utf8PathBuf> = vec![],



        /// Enables highlighting of related references while the cursor is on `break`, `loop`, `while`, or `for` keywords.
        highlightRelated_breakPoints_enable: bool = true,
        /// Enables highlighting of all captures of a closure while the cursor is on the `|` or move keyword of a closure.
        highlightRelated_closureCaptures_enable: bool = true,
        /// Enables highlighting of all exit points while the cursor is on any `return`, `?`, `fn`, or return type arrow (`->`).
        highlightRelated_exitPoints_enable: bool = true,
        /// Enables highlighting of related references while the cursor is on any identifier.
        highlightRelated_references_enable: bool = true,
        /// Enables highlighting of all break points for a loop or block context while the cursor is on any `async` or `await` keywords.
        highlightRelated_yieldPoints_enable: bool = true,

        /// Whether to show `Debug` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_debug_enable: bool           = true,
        /// Whether to show HoverActions in Rust files.
        hover_actions_enable: bool          = true,
        /// Whether to show `Go to Type Definition` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_gotoTypeDef_enable: bool     = true,
        /// Whether to show `Implementations` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_implementations_enable: bool = true,
        /// Whether to show `References` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_references_enable: bool      = false,
        /// Whether to show `Run` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` is set.
        hover_actions_run_enable: bool             = true,
        /// Whether to show `Update Test` action. Only applies when
        /// `#rust-analyzer.hover.actions.enable#` and `#rust-analyzer.hover.actions.run.enable#` are set.
        hover_actions_updateTest_enable: bool     = true,

        /// Whether to show documentation on hover.
        hover_documentation_enable: bool           = true,
        /// Whether to show keyword hover popups. Only applies when
        /// `#rust-analyzer.hover.documentation.enable#` is set.
        hover_documentation_keywords_enable: bool  = true,
        /// Whether to show drop glue information on hover.
        hover_dropGlue_enable: bool                = true,
        /// Use markdown syntax for links on hover.
        hover_links_enable: bool = true,
        /// Whether to show what types are used as generic arguments in calls etc. on hover, and what is their max length to show such types, beyond it they will be shown with ellipsis.
        ///
        /// This can take three values: `null` means "unlimited", the string `"hide"` means to not show generic substitutions at all, and a number means to limit them to X characters.
        ///
        /// The default is 20 characters.
        hover_maxSubstitutionLength: Option<MaxSubstitutionLength> = Some(MaxSubstitutionLength::Limit(20)),
        /// How to render the align information in a memory layout hover.
        hover_memoryLayout_alignment: Option<MemoryLayoutHoverRenderKindDef> = Some(MemoryLayoutHoverRenderKindDef::Hexadecimal),
        /// Whether to show memory layout data on hover.
        hover_memoryLayout_enable: bool = true,
        /// How to render the niche information in a memory layout hover.
        hover_memoryLayout_niches: Option<bool> = Some(false),
        /// How to render the offset information in a memory layout hover.
        hover_memoryLayout_offset: Option<MemoryLayoutHoverRenderKindDef> = Some(MemoryLayoutHoverRenderKindDef::Hexadecimal),
        /// How to render the padding information in a memory layout hover.
        hover_memoryLayout_padding: Option<MemoryLayoutHoverRenderKindDef> = None,
        /// How to render the size information in a memory layout hover.
        hover_memoryLayout_size: Option<MemoryLayoutHoverRenderKindDef> = Some(MemoryLayoutHoverRenderKindDef::Both),

        /// How many variants of an enum to display when hovering on. Show none if empty.
        hover_show_enumVariants: Option<usize> = Some(5),
        /// How many fields of a struct, variant or union to display when hovering on. Show none if empty.
        hover_show_fields: Option<usize> = Some(5),
        /// How many associated items of a trait to display when hovering a trait.
        hover_show_traitAssocItems: Option<usize> = None,

        /// Whether to show inlay type hints for binding modes.
        inlayHints_bindingModeHints_enable: bool                   = false,
        /// Whether to show inlay type hints for method chains.
        inlayHints_chainingHints_enable: bool                      = true,
        /// Whether to show inlay hints after a closing `}` to indicate what item it belongs to.
        inlayHints_closingBraceHints_enable: bool                  = true,
        /// Minimum number of lines required before the `}` until the hint is shown (set to 0 or 1
        /// to always show them).
        inlayHints_closingBraceHints_minLines: usize               = 25,
        /// Whether to show inlay hints for closure captures.
        inlayHints_closureCaptureHints_enable: bool                          = false,
        /// Whether to show inlay type hints for return types of closures.
        inlayHints_closureReturnTypeHints_enable: ClosureReturnTypeHintsDef  = ClosureReturnTypeHintsDef::Never,
        /// Closure notation in type and chaining inlay hints.
        inlayHints_closureStyle: ClosureStyle                                = ClosureStyle::ImplFn,
        /// Whether to show enum variant discriminant hints.
        inlayHints_discriminantHints_enable: DiscriminantHintsDef            = DiscriminantHintsDef::Never,
        /// Whether to show inlay hints for type adjustments.
        inlayHints_expressionAdjustmentHints_enable: AdjustmentHintsDef = AdjustmentHintsDef::Never,
        /// Whether to hide inlay hints for type adjustments outside of `unsafe` blocks.
        inlayHints_expressionAdjustmentHints_hideOutsideUnsafe: bool = false,
        /// Whether to show inlay hints as postfix ops (`.*` instead of `*`, etc).
        inlayHints_expressionAdjustmentHints_mode: AdjustmentHintsModeDef = AdjustmentHintsModeDef::Prefix,
        /// Whether to show const generic parameter name inlay hints.
        inlayHints_genericParameterHints_const_enable: bool= true,
        /// Whether to show generic lifetime parameter name inlay hints.
        inlayHints_genericParameterHints_lifetime_enable: bool = false,
        /// Whether to show generic type parameter name inlay hints.
        inlayHints_genericParameterHints_type_enable: bool = false,
        /// Whether to show implicit drop hints.
        inlayHints_implicitDrops_enable: bool                      = false,
        /// Whether to show inlay hints for the implied type parameter `Sized` bound.
        inlayHints_implicitSizedBoundHints_enable: bool            = false,
        /// Whether to show inlay type hints for elided lifetimes in function signatures.
        inlayHints_lifetimeElisionHints_enable: LifetimeElisionDef = LifetimeElisionDef::Never,
        /// Whether to prefer using parameter names as the name for elided lifetime hints if possible.
        inlayHints_lifetimeElisionHints_useParameterNames: bool    = false,
        /// Maximum length for inlay hints. Set to null to have an unlimited length.
        inlayHints_maxLength: Option<usize>                        = Some(25),
        /// Whether to show function parameter name inlay hints at the call
        /// site.
        inlayHints_parameterHints_enable: bool                     = true,
        /// Whether to show exclusive range inlay hints.
        inlayHints_rangeExclusiveHints_enable: bool                = false,
        /// Whether to show inlay hints for compiler inserted reborrows.
        /// This setting is deprecated in favor of #rust-analyzer.inlayHints.expressionAdjustmentHints.enable#.
        inlayHints_reborrowHints_enable: ReborrowHintsDef          = ReborrowHintsDef::Never,
        /// Whether to render leading colons for type hints, and trailing colons for parameter hints.
        inlayHints_renderColons: bool                              = true,
        /// Whether to show inlay type hints for variables.
        inlayHints_typeHints_enable: bool                          = true,
        /// Whether to hide inlay type hints for `let` statements that initialize to a closure.
        /// Only applies to closures with blocks, same as `#rust-analyzer.inlayHints.closureReturnTypeHints.enable#`.
        inlayHints_typeHints_hideClosureInitialization: bool       = false,
        /// Whether to hide inlay parameter type hints for closures.
        inlayHints_typeHints_hideClosureParameter:bool             = false,
        /// Whether to hide inlay type hints for constructors.
        inlayHints_typeHints_hideNamedConstructor: bool            = false,

        /// Enables the experimental support for interpreting tests.
        interpret_tests: bool = false,

        /// Join lines merges consecutive declaration and initialization of an assignment.
        joinLines_joinAssignments: bool = true,
        /// Join lines inserts else between consecutive ifs.
        joinLines_joinElseIf: bool = true,
        /// Join lines removes trailing commas.
        joinLines_removeTrailingComma: bool = true,
        /// Join lines unwraps trivial blocks.
        joinLines_unwrapTrivialBlock: bool = true,

        /// Whether to show `Debug` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_debug_enable: bool            = true,
        /// Whether to show CodeLens in Rust files.
        lens_enable: bool           = true,
        /// Whether to show `Implementations` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_implementations_enable: bool  = true,
        /// Where to render annotations.
        lens_location: AnnotationLocation = AnnotationLocation::AboveName,
        /// Whether to show `References` lens for Struct, Enum, and Union.
        /// Only applies when `#rust-analyzer.lens.enable#` is set.
        lens_references_adt_enable: bool = false,
        /// Whether to show `References` lens for Enum Variants.
        /// Only applies when `#rust-analyzer.lens.enable#` is set.
        lens_references_enumVariant_enable: bool = false,
        /// Whether to show `Method References` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_references_method_enable: bool = false,
        /// Whether to show `References` lens for Trait.
        /// Only applies when `#rust-analyzer.lens.enable#` is set.
        lens_references_trait_enable: bool = false,
        /// Whether to show `Run` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` is set.
        lens_run_enable: bool              = true,
        /// Whether to show `Update Test` lens. Only applies when
        /// `#rust-analyzer.lens.enable#` and `#rust-analyzer.lens.run.enable#` are set.
        lens_updateTest_enable: bool = true,

        /// Disable project auto-discovery in favor of explicitly specified set
        /// of projects.
        ///
        /// Elements must be paths pointing to `Cargo.toml`,
        /// `rust-project.json`, `.rs` files (which will be treated as standalone files) or JSON
        /// objects in `rust-project.json` format.
        linkedProjects: Vec<ManifestOrProjectJson> = vec![],

        /// Number of syntax trees rust-analyzer keeps in memory. Defaults to 128.
        lru_capacity: Option<u16>                 = None,
        /// Sets the LRU capacity of the specified queries.
        lru_query_capacities: FxHashMap<Box<str>, u16> = FxHashMap::default(),

        /// Whether to show `can't find Cargo.toml` error message.
        notifications_cargoTomlNotFound: bool      = true,

        /// How many worker threads in the main loop. The default `null` means to pick automatically.
        numThreads: Option<NumThreads> = None,

        /// Expand attribute macros. Requires `#rust-analyzer.procMacro.enable#` to be set.
        procMacro_attributes_enable: bool = true,
        /// Enable support for procedural macros, implies `#rust-analyzer.cargo.buildScripts.enable#`.
        procMacro_enable: bool                     = true,
        /// Internal config, path to proc-macro server executable.
        procMacro_server: Option<Utf8PathBuf>          = None,

        /// Exclude imports from find-all-references.
        references_excludeImports: bool = false,

        /// Exclude tests from find-all-references and call-hierarchy.
        references_excludeTests: bool = false,

        /// Inject additional highlighting into doc comments.
        ///
        /// When enabled, rust-analyzer will highlight rust source in doc comments as well as intra
        /// doc links.
        semanticHighlighting_doc_comment_inject_enable: bool = true,
        /// Whether the server is allowed to emit non-standard tokens and modifiers.
        semanticHighlighting_nonStandardTokens: bool = true,
        /// Use semantic tokens for operators.
        ///
        /// When disabled, rust-analyzer will emit semantic tokens only for operator tokens when
        /// they are tagged with modifiers.
        semanticHighlighting_operator_enable: bool = true,
        /// Use specialized semantic tokens for operators.
        ///
        /// When enabled, rust-analyzer will emit special token types for operator tokens instead
        /// of the generic `operator` token type.
        semanticHighlighting_operator_specialization_enable: bool = false,
        /// Use semantic tokens for punctuation.
        ///
        /// When disabled, rust-analyzer will emit semantic tokens only for punctuation tokens when
        /// they are tagged with modifiers or have a special role.
        semanticHighlighting_punctuation_enable: bool = false,
        /// When enabled, rust-analyzer will emit a punctuation semantic token for the `!` of macro
        /// calls.
        semanticHighlighting_punctuation_separate_macro_bang: bool = false,
        /// Use specialized semantic tokens for punctuation.
        ///
        /// When enabled, rust-analyzer will emit special token types for punctuation tokens instead
        /// of the generic `punctuation` token type.
        semanticHighlighting_punctuation_specialization_enable: bool = false,
        /// Use semantic tokens for strings.
        ///
        /// In some editors (e.g. vscode) semantic tokens override other highlighting grammars.
        /// By disabling semantic tokens for strings, other grammars can be used to highlight
        /// their contents.
        semanticHighlighting_strings_enable: bool = true,

        /// Show full signature of the callable. Only shows parameters if disabled.
        signatureInfo_detail: SignatureDetail                           = SignatureDetail::Full,
        /// Show documentation.
        signatureInfo_documentation_enable: bool                       = true,

        /// Specify the characters allowed to invoke special on typing triggers.
        /// - typing `=` after `let` tries to smartly add `;` if `=` is followed by an existing expression
        /// - typing `=` between two expressions adds `;` when in statement position
        /// - typing `=` to turn an assignment into an equality comparison removes `;` when in expression position
        /// - typing `.` in a chain method call auto-indents
        /// - typing `{` or `(` in front of an expression inserts a closing `}` or `)` after the expression
        /// - typing `{` in a use item adds a closing `}` in the right place
        /// - typing `>` to complete a return type `->` will insert a whitespace after it
        /// - typing `<` in a path or type position inserts a closing `>` after the path or type.
        typing_triggerChars: Option<String> = Some("=.".to_owned()),


        /// Enables automatic discovery of projects using [`DiscoverWorkspaceConfig::command`].
        ///
        /// [`DiscoverWorkspaceConfig`] also requires setting `progress_label` and `files_to_watch`.
        /// `progress_label` is used for the title in progress indicators, whereas `files_to_watch`
        /// is used to determine which build system-specific files should be watched in order to
        /// reload rust-analyzer.
        ///
        /// Below is an example of a valid configuration:
        /// ```json
        /// "rust-analyzer.workspace.discoverConfig": {
        ///     "command": [
        ///         "rust-project",
        ///         "develop-json"
        ///     ],
        ///     "progressLabel": "rust-analyzer",
        ///     "filesToWatch": [
        ///         "BUCK"
        ///     ]
        /// }
        /// ```
        ///
        /// ## On `DiscoverWorkspaceConfig::command`
        ///
        /// **Warning**: This format is provisional and subject to change.
        ///
        /// [`DiscoverWorkspaceConfig::command`] *must* return a JSON object
        /// corresponding to `DiscoverProjectData::Finished`:
        ///
        /// ```norun
        /// #[derive(Debug, Clone, Deserialize, Serialize)]
        /// #[serde(tag = "kind")]
        /// #[serde(rename_all = "snake_case")]
        /// enum DiscoverProjectData {
        ///     Finished { buildfile: Utf8PathBuf, project: ProjectJsonData },
        ///     Error { error: String, source: Option<String> },
        ///     Progress { message: String },
        /// }
        /// ```
        ///
        /// As JSON, `DiscoverProjectData::Finished` is:
        ///
        /// ```json
        /// {
        ///     // the internally-tagged representation of the enum.
        ///     "kind": "finished",
        ///     // the file used by a non-Cargo build system to define
        ///     // a package or target.
        ///     "buildfile": "rust-analyzer/BUILD",
        ///     // the contents of a rust-project.json, elided for brevity
        ///     "project": {
        ///         "sysroot": "foo",
        ///         "crates": []
        ///     }
        /// }
        /// ```
        ///
        /// It is encouraged, but not required, to use the other variants on
        /// `DiscoverProjectData` to provide a more polished end-user experience.
        ///
        /// `DiscoverWorkspaceConfig::command` may *optionally* include an `{arg}`,
        /// which will be substituted with the JSON-serialized form of the following
        /// enum:
        ///
        /// ```norun
        /// #[derive(PartialEq, Clone, Debug, Serialize)]
        /// #[serde(rename_all = "camelCase")]
        /// pub enum DiscoverArgument {
        ///    Path(AbsPathBuf),
        ///    Buildfile(AbsPathBuf),
        /// }
        /// ```
        ///
        /// The JSON representation of `DiscoverArgument::Path` is:
        ///
        /// ```json
        /// {
        ///     "path": "src/main.rs"
        /// }
        /// ```
        ///
        /// Similarly, the JSON representation of `DiscoverArgument::Buildfile` is:
        ///
        /// ```json
        /// {
        ///     "buildfile": "BUILD"
        /// }
        /// ```
        ///
        /// `DiscoverArgument::Path` is used to find and generate a `rust-project.json`,
        /// and therefore, a workspace, whereas `DiscoverArgument::buildfile` is used to
        /// to update an existing workspace. As a reference for implementors,
        /// buck2's `rust-project` will likely be useful:
        /// https://github.com/facebook/buck2/tree/main/integrations/rust-project.
        workspace_discoverConfig: Option<DiscoverWorkspaceConfig> = None,
    }
}

config_data! {
    /// Local configurations can be defined per `SourceRoot`. This almost always corresponds to a `Crate`.
    local: struct LocalDefaultConfigData <- LocalConfigInput ->  {
        /// Whether to insert #[must_use] when generating `as_` methods
        /// for enum variants.
        assist_emitMustUse: bool               = false,
        /// Placeholder expression to use for missing expressions in assists.
        assist_expressionFillDefault: ExprFillDefaultDef              = ExprFillDefaultDef::Todo,
        /// Enable borrow checking for term search code assists. If set to false, also there will be more suggestions, but some of them may not borrow-check.
        assist_termSearch_borrowcheck: bool = true,
        /// Term search fuel in "units of work" for assists (Defaults to 1800).
        assist_termSearch_fuel: usize = 1800,


        /// Whether to automatically add a semicolon when completing unit-returning functions.
        ///
        /// In `match` arms it completes a comma instead.
        completion_addSemicolonToUnit: bool = true,
        /// Toggles the additional completions that automatically show method calls and field accesses with `await` prefixed to them when completing on a future.
        completion_autoAwait_enable: bool        = true,
        /// Toggles the additional completions that automatically show method calls with `iter()` or `into_iter()` prefixed to them when completing on a type that has them.
        completion_autoIter_enable: bool        = true,
        /// Toggles the additional completions that automatically add imports when completed.
        /// Note that your client must specify the `additionalTextEdits` LSP client capability to truly have this feature enabled.
        completion_autoimport_enable: bool       = true,
        /// A list of full paths to items to exclude from auto-importing completions.
        ///
        /// Traits in this list won't have their methods suggested in completions unless the trait
        /// is in scope.
        ///
        /// You can either specify a string path which defaults to type "always" or use the more verbose
        /// form `{ "path": "path::to::item", type: "always" }`.
        ///
        /// For traits the type "methods" can be used to only exclude the methods but not the trait itself.
        ///
        /// This setting also inherits `#rust-analyzer.completion.excludeTraits#`.
        completion_autoimport_exclude: Vec<AutoImportExclusion> = vec![
            AutoImportExclusion::Verbose { path: "core::borrow::Borrow".to_owned(), r#type: AutoImportExclusionType::Methods },
            AutoImportExclusion::Verbose { path: "core::borrow::BorrowMut".to_owned(), r#type: AutoImportExclusionType::Methods },
        ],
        /// Toggles the additional completions that automatically show method calls and field accesses
        /// with `self` prefixed to them when inside a method.
        completion_autoself_enable: bool        = true,
        /// Whether to add parenthesis and argument snippets when completing function.
        completion_callable_snippets: CallableCompletionDef  = CallableCompletionDef::FillArguments,
        /// A list of full paths to traits whose methods to exclude from completion.
        ///
        /// Methods from these traits won't be completed, even if the trait is in scope. However, they will still be suggested on expressions whose type is `dyn Trait`, `impl Trait` or `T where T: Trait`.
        ///
        /// Note that the trait themselves can still be completed.
        completion_excludeTraits: Vec<String> = Vec::new(),
        /// Whether to show full function/method signatures in completion docs.
        completion_fullFunctionSignatures_enable: bool = false,
        /// Whether to omit deprecated items from autocompletion. By default they are marked as deprecated but not hidden.
        completion_hideDeprecated: bool = false,
        /// Maximum number of completions to return. If `None`, the limit is infinite.
        completion_limit: Option<usize> = None,
        /// Whether to show postfix snippets like `dbg`, `if`, `not`, etc.
        completion_postfix_enable: bool         = true,
        /// Enables completions of private items and fields that are defined in the current workspace even if they are not visible at the current position.
        completion_privateEditable_enable: bool = false,
        /// Whether to enable term search based snippets like `Some(foo.bar().baz())`.
        completion_termSearch_enable: bool = false,
        /// Term search fuel in "units of work" for autocompletion (Defaults to 1000).
        completion_termSearch_fuel: usize = 1000,

        /// List of rust-analyzer diagnostics to disable.
        diagnostics_disabled: FxHashSet<String> = FxHashSet::default(),
        /// Whether to show native rust-analyzer diagnostics.
        diagnostics_enable: bool                = true,
        /// Whether to show experimental rust-analyzer diagnostics that might
        /// have more false positives than usual.
        diagnostics_experimental_enable: bool    = false,
        /// Map of prefixes to be substituted when parsing diagnostic file paths.
        /// This should be the reverse mapping of what is passed to `rustc` as `--remap-path-prefix`.
        diagnostics_remapPrefix: FxHashMap<String, String> = FxHashMap::default(),
        /// Whether to run additional style lints.
        diagnostics_styleLints_enable: bool =    false,
        /// List of warnings that should be displayed with hint severity.
        ///
        /// The warnings will be indicated by faded text or three dots in code
        /// and will not show up in the `Problems Panel`.
        diagnostics_warningsAsHint: Vec<String> = vec![],
        /// List of warnings that should be displayed with info severity.
        ///
        /// The warnings will be indicated by a blue squiggly underline in code
        /// and a blue icon in the `Problems Panel`.
        diagnostics_warningsAsInfo: Vec<String> = vec![],

        /// Whether to enforce the import granularity setting for all files. If set to false rust-analyzer will try to keep import styles consistent per file.
        imports_granularity_enforce: bool              = false,
        /// How imports should be grouped into use statements.
        imports_granularity_group: ImportGranularityDef  = ImportGranularityDef::Crate,
        /// Group inserted imports by the [following order](https://rust-analyzer.github.io/book/features.html#auto-import). Groups are separated by newlines.
        imports_group_enable: bool                           = true,
        /// Whether to allow import insertion to merge new imports into single path glob imports like `use std::fmt::*;`.
        imports_merge_glob: bool           = true,
        /// Prefer to unconditionally use imports of the core and alloc crate, over the std crate.
        imports_preferNoStd | imports_prefer_no_std: bool = false,
         /// Whether to prefer import paths containing a `prelude` module.
        imports_preferPrelude: bool                       = false,
        /// The path structure for newly inserted paths to use.
        imports_prefix: ImportPrefixDef               = ImportPrefixDef::ByCrate,
        /// Whether to prefix external (including std, core) crate imports with `::`. e.g. "use ::std::io::Read;".
        imports_prefixExternPrelude: bool = false,
    }
}

config_data! {
    workspace: struct WorkspaceDefaultConfigData <- WorkspaceConfigInput -> {
        /// Pass `--all-targets` to cargo invocation.
        cargo_allTargets: bool           = true,
        /// Automatically refresh project info via `cargo metadata` on
        /// `Cargo.toml` or `.cargo/config.toml` changes.
        cargo_autoreload: bool           = true,
        /// Run build scripts (`build.rs`) for more precise code analysis.
        cargo_buildScripts_enable: bool  = true,
        /// Specifies the invocation strategy to use when running the build scripts command.
        /// If `per_workspace` is set, the command will be executed for each Rust workspace with the
        /// workspace as the working directory.
        /// If `once` is set, the command will be executed once with the opened project as the
        /// working directory.
        /// This config only has an effect when `#rust-analyzer.cargo.buildScripts.overrideCommand#`
        /// is set.
        cargo_buildScripts_invocationStrategy: InvocationStrategy = InvocationStrategy::PerWorkspace,
        /// Override the command rust-analyzer uses to run build scripts and
        /// build procedural macros. The command is required to output json
        /// and should therefore include `--message-format=json` or a similar
        /// option.
        ///
        /// If there are multiple linked projects/workspaces, this command is invoked for
        /// each of them, with the working directory being the workspace root
        /// (i.e., the folder containing the `Cargo.toml`). This can be overwritten
        /// by changing `#rust-analyzer.cargo.buildScripts.invocationStrategy#`.
        ///
        /// By default, a cargo invocation will be constructed for the configured
        /// targets and features, with the following base command line:
        ///
        /// ```bash
        /// cargo check --quiet --workspace --message-format=json --all-targets --keep-going
        /// ```
        /// .
        cargo_buildScripts_overrideCommand: Option<Vec<String>> = None,
        /// Rerun proc-macros building/build-scripts running when proc-macro
        /// or build-script sources change and are saved.
        cargo_buildScripts_rebuildOnSave: bool = true,
        /// Use `RUSTC_WRAPPER=rust-analyzer` when running build scripts to
        /// avoid checking unnecessary things.
        cargo_buildScripts_useRustcWrapper: bool = true,
        /// List of cfg options to enable with the given values.
        ///
        /// To enable a name without a value, use `"key"`.
        /// To enable a name with a value, use `"key=value"`.
        /// To disable, prefix the entry with a `!`.
        cargo_cfgs: Vec<String> = {
            vec!["debug_assertions".into(), "miri".into()]
        },
        /// Extra arguments that are passed to every cargo invocation.
        cargo_extraArgs: Vec<String> = vec![],
        /// Extra environment variables that will be set when running cargo, rustc
        /// or other commands within the workspace. Useful for setting RUSTFLAGS.
        cargo_extraEnv: FxHashMap<String, Option<String>> = FxHashMap::default(),
        /// List of features to activate.
        ///
        /// Set this to `"all"` to pass `--all-features` to cargo.
        cargo_features: CargoFeaturesDef      = CargoFeaturesDef::Selected(vec![]),
        /// Whether to pass `--no-default-features` to cargo.
        cargo_noDefaultFeatures: bool    = false,
        /// Whether to skip fetching dependencies. If set to "true", the analysis is performed
        /// entirely offline, and Cargo metadata for dependencies is not fetched.
        cargo_noDeps: bool = false,
        /// Relative path to the sysroot, or "discover" to try to automatically find it via
        /// "rustc --print sysroot".
        ///
        /// Unsetting this disables sysroot loading.
        ///
        /// This option does not take effect until rust-analyzer is restarted.
        cargo_sysroot: Option<String>    = Some("discover".to_owned()),
        /// Relative path to the sysroot library sources. If left unset, this will default to
        /// `{cargo.sysroot}/lib/rustlib/src/rust/library`.
        ///
        /// This option does not take effect until rust-analyzer is restarted.
        cargo_sysrootSrc: Option<String>    = None,
        /// Compilation target override (target tuple).
        // FIXME(@poliorcetics): move to multiple targets here too, but this will need more work
        // than `checkOnSave_target`
        cargo_target: Option<String>     = None,
        /// Optional path to a rust-analyzer specific target directory.
        /// This prevents rust-analyzer's `cargo check` and initial build-script and proc-macro
        /// building from locking the `Cargo.lock` at the expense of duplicating build artifacts.
        ///
        /// Set to `true` to use a subdirectory of the existing target directory or
        /// set to a path relative to the workspace to use that path.
        cargo_targetDir | rust_analyzerTargetDir: Option<TargetDirectory> = None,

        /// Set `cfg(test)` for local crates. Defaults to true.
        cfg_setTest: bool = true,

        /// Run the check command for diagnostics on save.
        checkOnSave | checkOnSave_enable: bool                         = true,


        /// Check all targets and tests (`--all-targets`). Defaults to
        /// `#rust-analyzer.cargo.allTargets#`.
        check_allTargets | checkOnSave_allTargets: Option<bool>          = None,
        /// Cargo command to use for `cargo check`.
        check_command | checkOnSave_command: String                      = "check".to_owned(),
        /// Extra arguments for `cargo check`.
        check_extraArgs | checkOnSave_extraArgs: Vec<String>             = vec![],
        /// Extra environment variables that will be set when running `cargo check`.
        /// Extends `#rust-analyzer.cargo.extraEnv#`.
        check_extraEnv | checkOnSave_extraEnv: FxHashMap<String, Option<String>> = FxHashMap::default(),
        /// List of features to activate. Defaults to
        /// `#rust-analyzer.cargo.features#`.
        ///
        /// Set to `"all"` to pass `--all-features` to Cargo.
        check_features | checkOnSave_features: Option<CargoFeaturesDef>  = None,
        /// List of `cargo check` (or other command specified in `check.command`) diagnostics to ignore.
        ///
        /// For example for `cargo check`: `dead_code`, `unused_imports`, `unused_variables`,...
        check_ignore: FxHashSet<String> = FxHashSet::default(),
        /// Specifies the invocation strategy to use when running the check command.
        /// If `per_workspace` is set, the command will be executed for each workspace.
        /// If `once` is set, the command will be executed once.
        /// This config only has an effect when `#rust-analyzer.check.overrideCommand#`
        /// is set.
        check_invocationStrategy | checkOnSave_invocationStrategy: InvocationStrategy = InvocationStrategy::PerWorkspace,
        /// Whether to pass `--no-default-features` to Cargo. Defaults to
        /// `#rust-analyzer.cargo.noDefaultFeatures#`.
        check_noDefaultFeatures | checkOnSave_noDefaultFeatures: Option<bool>         = None,
        /// Override the command rust-analyzer uses instead of `cargo check` for
        /// diagnostics on save. The command is required to output json and
        /// should therefore include `--message-format=json` or a similar option
        /// (if your client supports the `colorDiagnosticOutput` experimental
        /// capability, you can use `--message-format=json-diagnostic-rendered-ansi`).
        ///
        /// If you're changing this because you're using some tool wrapping
        /// Cargo, you might also want to change
        /// `#rust-analyzer.cargo.buildScripts.overrideCommand#`.
        ///
        /// If there are multiple linked projects/workspaces, this command is invoked for
        /// each of them, with the working directory being the workspace root
        /// (i.e., the folder containing the `Cargo.toml`). This can be overwritten
        /// by changing `#rust-analyzer.check.invocationStrategy#`.
        ///
        /// If `$saved_file` is part of the command, rust-analyzer will pass
        /// the absolute path of the saved file to the provided command. This is
        /// intended to be used with non-Cargo build systems.
        /// Note that `$saved_file` is experimental and may be removed in the future.
        ///
        /// An example command would be:
        ///
        /// ```bash
        /// cargo check --workspace --message-format=json --all-targets
        /// ```
        /// .
        check_overrideCommand | checkOnSave_overrideCommand: Option<Vec<String>>             = None,
        /// Check for specific targets. Defaults to `#rust-analyzer.cargo.target#` if empty.
        ///
        /// Can be a single target, e.g. `"x86_64-unknown-linux-gnu"` or a list of targets, e.g.
        /// `["aarch64-apple-darwin", "x86_64-apple-darwin"]`.
        ///
        /// Aliased as `"checkOnSave.targets"`.
        check_targets | checkOnSave_targets | checkOnSave_target: Option<CheckOnSaveTargets> = None,
        /// Whether `--workspace` should be passed to `cargo check`.
        /// If false, `-p <package>` will be passed instead if applicable. In case it is not, no
        /// check will be performed.
        check_workspace: bool = true,

        /// These proc-macros will be ignored when trying to expand them.
        ///
        /// This config takes a map of crate names with the exported proc-macro names to ignore as values.
        procMacro_ignored: FxHashMap<Box<str>, Box<[Box<str>]>>          = FxHashMap::default(),

        /// Command to be executed instead of 'cargo' for runnables.
        runnables_command: Option<String> = None,
        /// Additional arguments to be passed to cargo for runnables such as
        /// tests or binaries. For example, it may be `--release`.
        runnables_extraArgs: Vec<String>   = vec![],
        /// Additional arguments to be passed through Cargo to launched tests, benchmarks, or
        /// doc-tests.
        ///
        /// Unless the launched target uses a
        /// [custom test harness](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#the-harness-field),
        /// they will end up being interpreted as options to
        /// [`rustc`’s built-in test harness (“libtest”)](https://doc.rust-lang.org/rustc/tests/index.html#cli-arguments).
        runnables_extraTestBinaryArgs: Vec<String> = vec!["--show-output".to_owned()],

        /// Path to the Cargo.toml of the rust compiler workspace, for usage in rustc_private
        /// projects, or "discover" to try to automatically find it if the `rustc-dev` component
        /// is installed.
        ///
        /// Any project which uses rust-analyzer with the rustcPrivate
        /// crates must set `[package.metadata.rust-analyzer] rustc_private=true` to use it.
        ///
        /// This option does not take effect until rust-analyzer is restarted.
        rustc_source: Option<String> = None,

        /// Additional arguments to `rustfmt`.
        rustfmt_extraArgs: Vec<String>               = vec![],
        /// Advanced option, fully override the command rust-analyzer uses for
        /// formatting. This should be the equivalent of `rustfmt` here, and
        /// not that of `cargo fmt`. The file contents will be passed on the
        /// standard input and the formatted result will be read from the
        /// standard output.
        rustfmt_overrideCommand: Option<Vec<String>> = None,
        /// Enables the use of rustfmt's unstable range formatting command for the
        /// `textDocument/rangeFormatting` request. The rustfmt option is unstable and only
        /// available on a nightly build.
        rustfmt_rangeFormatting_enable: bool = false,

        /// Additional paths to include in the VFS. Generally for code that is
        /// generated or otherwise managed by a build system outside of Cargo,
        /// though Cargo might be the eventual consumer.
        vfs_extraIncludes: Vec<String> = vec![],

        /// Workspace symbol search kind.
        workspace_symbol_search_kind: WorkspaceSymbolSearchKindDef = WorkspaceSymbolSearchKindDef::OnlyTypes,
        /// Limits the number of items returned from a workspace symbol search (Defaults to 128).
        /// Some clients like vs-code issue new searches on result filtering and don't require all results to be returned in the initial search.
        /// Other clients requires all results upfront and might require a higher limit.
        workspace_symbol_search_limit: usize = 128,
        /// Workspace symbol search scope.
        workspace_symbol_search_scope: WorkspaceSymbolSearchScopeDef = WorkspaceSymbolSearchScopeDef::Workspace,
    }
}

config_data! {
    /// Configs that only make sense when they are set by a client. As such they can only be defined
    /// by setting them using client's settings (e.g `settings.json` on VS Code).
    client: struct ClientDefaultConfigData <- ClientConfigInput -> {

        /// Controls file watching implementation.
        files_watcher: FilesWatcherDef = FilesWatcherDef::Client,


    }
}

#[derive(Debug)]
pub enum RatomlFileKind {
    Workspace,
    Crate,
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum RatomlFile {
    Workspace(WorkspaceLocalConfigInput),
    Crate(LocalConfigInput),
}

#[derive(Clone, Debug)]
struct ClientInfo {
    name: String,
    version: Option<Version>,
}

#[derive(Clone)]
pub struct Config {
    /// Projects that have a Cargo.toml or a rust-project.json in a
    /// parent directory, so we can discover them by walking the
    /// file system.
    discovered_projects_from_filesystem: Vec<ProjectManifest>,
    /// Projects whose configuration was generated by a command
    /// configured in discoverConfig.
    discovered_projects_from_command: Vec<ProjectJsonFromCommand>,
    /// The workspace roots as registered by the LSP client
    workspace_roots: Vec<AbsPathBuf>,
    caps: ClientCapabilities,
    root_path: AbsPathBuf,
    snippets: Vec<Snippet>,
    client_info: Option<ClientInfo>,

    default_config: &'static DefaultConfigData,
    /// Config node that obtains its initial value during the server initialization and
    /// by receiving a `lsp_types::notification::DidChangeConfiguration`.
    client_config: (FullConfigInput, ConfigErrors),

    /// Config node whose values apply to **every** Rust project.
    user_config: Option<(GlobalWorkspaceLocalConfigInput, ConfigErrors)>,

    ratoml_file: FxHashMap<SourceRootId, (RatomlFile, ConfigErrors)>,

    /// Clone of the value that is stored inside a `GlobalState`.
    source_root_parent_map: Arc<FxHashMap<SourceRootId, SourceRootId>>,

    /// Use case : It is an error to have an empty value for `check_command`.
    /// Since it is a `global` command at the moment, its final value can only be determined by
    /// traversing through `global` configs and the `client` config. However the non-null value constraint
    /// is config level agnostic, so this requires an independent error storage
    validation_errors: ConfigErrors,

    detached_files: Vec<AbsPathBuf>,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("discovered_projects_from_filesystem", &self.discovered_projects_from_filesystem)
            .field("discovered_projects_from_command", &self.discovered_projects_from_command)
            .field("workspace_roots", &self.workspace_roots)
            .field("caps", &self.caps)
            .field("root_path", &self.root_path)
            .field("snippets", &self.snippets)
            .field("client_info", &self.client_info)
            .field("client_config", &self.client_config)
            .field("user_config", &self.user_config)
            .field("ratoml_file", &self.ratoml_file)
            .field("source_root_parent_map", &self.source_root_parent_map)
            .field("validation_errors", &self.validation_errors)
            .field("detached_files", &self.detached_files)
            .finish()
    }
}

// Delegate capability fetching methods
impl std::ops::Deref for Config {
    type Target = ClientCapabilities;

    fn deref(&self) -> &Self::Target {
        &self.caps
    }
}

impl Config {
    /// Path to the user configuration dir. This can be seen as a generic way to define what would be `$XDG_CONFIG_HOME/rust-analyzer` in Linux.
    pub fn user_config_dir_path() -> Option<AbsPathBuf> {
        let user_config_path = if let Some(path) = env::var_os("__TEST_RA_USER_CONFIG_DIR") {
            std::path::PathBuf::from(path)
        } else {
            dirs::config_dir()?.join("rust-analyzer")
        };
        Some(AbsPathBuf::assert_utf8(user_config_path))
    }

    pub fn same_source_root_parent_map(
        &self,
        other: &Arc<FxHashMap<SourceRootId, SourceRootId>>,
    ) -> bool {
        Arc::ptr_eq(&self.source_root_parent_map, other)
    }

    // FIXME @alibektas : Server's health uses error sink but in other places it is not used atm.
    /// Changes made to client and global configurations will partially not be reflected even after `.apply_change()` was called.
    /// The return tuple's bool component signals whether the `GlobalState` should call its `update_configuration()` method.
    fn apply_change_with_sink(&self, change: ConfigChange) -> (Config, bool) {
        let mut config = self.clone();
        config.validation_errors = ConfigErrors::default();

        let mut should_update = false;

        if let Some(change) = change.user_config_change {
            tracing::info!("updating config from user config toml: {:#}", change);
            if let Ok(table) = toml::from_str(&change) {
                let mut toml_errors = vec![];
                validate_toml_table(
                    GlobalWorkspaceLocalConfigInput::FIELDS,
                    &table,
                    &mut String::new(),
                    &mut toml_errors,
                );
                config.user_config = Some((
                    GlobalWorkspaceLocalConfigInput::from_toml(table, &mut toml_errors),
                    ConfigErrors(
                        toml_errors
                            .into_iter()
                            .map(|(a, b)| ConfigErrorInner::Toml { config_key: a, error: b })
                            .map(Arc::new)
                            .collect(),
                    ),
                ));
                should_update = true;
            }
        }

        if let Some(mut json) = change.client_config_change {
            tracing::info!("updating config from JSON: {:#}", json);

            if !(json.is_null() || json.as_object().is_some_and(|it| it.is_empty())) {
                let detached_files = get_field_json::<Vec<Utf8PathBuf>>(
                    &mut json,
                    &mut Vec::new(),
                    "detachedFiles",
                    None,
                )
                .unwrap_or_default()
                .into_iter()
                .map(AbsPathBuf::assert)
                .collect();

                patch_old_style::patch_json_for_outdated_configs(&mut json);

                let mut json_errors = vec![];

                let input = FullConfigInput::from_json(json, &mut json_errors);

                // IMPORTANT : This holds as long as ` completion_snippets_custom` is declared `client`.
                config.snippets.clear();

                let snips = input
                    .global
                    .completion_snippets_custom
                    .as_ref()
                    .unwrap_or(&self.default_config.global.completion_snippets_custom);
                #[allow(dead_code)]
                let _ = Self::completion_snippets_custom;
                for (name, def) in snips.iter() {
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
                        Some(snippet) => config.snippets.push(snippet),
                        None => json_errors.push((
                            name.to_owned(),
                            <serde_json::Error as serde::de::Error>::custom(format!(
                                "snippet {name} is invalid or triggers are missing",
                            )),
                        )),
                    }
                }

                config.client_config = (
                    input,
                    ConfigErrors(
                        json_errors
                            .into_iter()
                            .map(|(a, b)| ConfigErrorInner::Json { config_key: a, error: b })
                            .map(Arc::new)
                            .collect(),
                    ),
                );
                config.detached_files = detached_files;
            }
            should_update = true;
        }

        if let Some(change) = change.ratoml_file_change {
            for (source_root_id, (kind, _, text)) in change {
                match kind {
                    RatomlFileKind::Crate => {
                        if let Some(text) = text {
                            let mut toml_errors = vec![];
                            tracing::info!("updating ra-toml crate config: {:#}", text);
                            match toml::from_str(&text) {
                                Ok(table) => {
                                    validate_toml_table(
                                        &[LocalConfigInput::FIELDS],
                                        &table,
                                        &mut String::new(),
                                        &mut toml_errors,
                                    );
                                    config.ratoml_file.insert(
                                        source_root_id,
                                        (
                                            RatomlFile::Crate(LocalConfigInput::from_toml(
                                                &table,
                                                &mut toml_errors,
                                            )),
                                            ConfigErrors(
                                                toml_errors
                                                    .into_iter()
                                                    .map(|(a, b)| ConfigErrorInner::Toml {
                                                        config_key: a,
                                                        error: b,
                                                    })
                                                    .map(Arc::new)
                                                    .collect(),
                                            ),
                                        ),
                                    );
                                }
                                Err(e) => {
                                    config.validation_errors.0.push(
                                        ConfigErrorInner::ParseError {
                                            reason: e.message().to_owned(),
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }
                    RatomlFileKind::Workspace => {
                        if let Some(text) = text {
                            tracing::info!("updating ra-toml workspace config: {:#}", text);
                            let mut toml_errors = vec![];
                            match toml::from_str(&text) {
                                Ok(table) => {
                                    validate_toml_table(
                                        WorkspaceLocalConfigInput::FIELDS,
                                        &table,
                                        &mut String::new(),
                                        &mut toml_errors,
                                    );
                                    config.ratoml_file.insert(
                                        source_root_id,
                                        (
                                            RatomlFile::Workspace(
                                                WorkspaceLocalConfigInput::from_toml(
                                                    table,
                                                    &mut toml_errors,
                                                ),
                                            ),
                                            ConfigErrors(
                                                toml_errors
                                                    .into_iter()
                                                    .map(|(a, b)| ConfigErrorInner::Toml {
                                                        config_key: a,
                                                        error: b,
                                                    })
                                                    .map(Arc::new)
                                                    .collect(),
                                            ),
                                        ),
                                    );
                                    should_update = true;
                                }
                                Err(e) => {
                                    config.validation_errors.0.push(
                                        ConfigErrorInner::ParseError {
                                            reason: e.message().to_owned(),
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(source_root_map) = change.source_map_change {
            config.source_root_parent_map = source_root_map;
        }

        if config.check_command(None).is_empty() {
            config.validation_errors.0.push(Arc::new(ConfigErrorInner::Json {
                config_key: "/check/command".to_owned(),
                error: serde_json::Error::custom("expected a non-empty string"),
            }));
        }

        (config, should_update)
    }

    /// Given `change` this generates a new `Config`, thereby collecting errors of type `ConfigError`.
    /// If there are changes that have global/client level effect, the last component of the return type
    /// will be set to `true`, which should be used by the `GlobalState` to update itself.
    pub fn apply_change(&self, change: ConfigChange) -> (Config, ConfigErrors, bool) {
        let (config, should_update) = self.apply_change_with_sink(change);
        let e = ConfigErrors(
            config
                .client_config
                .1
                .0
                .iter()
                .chain(config.user_config.as_ref().into_iter().flat_map(|it| it.1.0.iter()))
                .chain(config.ratoml_file.values().flat_map(|it| it.1.0.iter()))
                .chain(config.validation_errors.0.iter())
                .cloned()
                .collect(),
        );
        (config, e, should_update)
    }

    pub fn add_discovered_project_from_command(
        &mut self,
        data: ProjectJsonData,
        buildfile: AbsPathBuf,
    ) {
        for proj in self.discovered_projects_from_command.iter_mut() {
            if proj.buildfile == buildfile {
                proj.data = data;
                return;
            }
        }

        self.discovered_projects_from_command.push(ProjectJsonFromCommand { data, buildfile });
    }
}

#[derive(Default, Debug)]
pub struct ConfigChange {
    user_config_change: Option<Arc<str>>,
    client_config_change: Option<serde_json::Value>,
    ratoml_file_change:
        Option<FxHashMap<SourceRootId, (RatomlFileKind, VfsPath, Option<Arc<str>>)>>,
    source_map_change: Option<Arc<FxHashMap<SourceRootId, SourceRootId>>>,
}

impl ConfigChange {
    pub fn change_ratoml(
        &mut self,
        source_root: SourceRootId,
        vfs_path: VfsPath,
        content: Option<Arc<str>>,
    ) -> Option<(RatomlFileKind, VfsPath, Option<Arc<str>>)> {
        self.ratoml_file_change
            .get_or_insert_with(Default::default)
            .insert(source_root, (RatomlFileKind::Crate, vfs_path, content))
    }

    pub fn change_user_config(&mut self, content: Option<Arc<str>>) {
        assert!(self.user_config_change.is_none()); // Otherwise it is a double write.
        self.user_config_change = content;
    }

    pub fn change_workspace_ratoml(
        &mut self,
        source_root: SourceRootId,
        vfs_path: VfsPath,
        content: Option<Arc<str>>,
    ) -> Option<(RatomlFileKind, VfsPath, Option<Arc<str>>)> {
        self.ratoml_file_change
            .get_or_insert_with(Default::default)
            .insert(source_root, (RatomlFileKind::Workspace, vfs_path, content))
    }

    pub fn change_client_config(&mut self, change: serde_json::Value) {
        self.client_config_change = Some(change);
    }

    pub fn change_source_root_parent_map(
        &mut self,
        source_root_map: Arc<FxHashMap<SourceRootId, SourceRootId>>,
    ) {
        assert!(self.source_map_change.is_none());
        self.source_map_change = Some(source_root_map);
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LinkedProject {
    ProjectManifest(ProjectManifest),
    InlineProjectJson(ProjectJson),
}

impl From<ProjectManifest> for LinkedProject {
    fn from(v: ProjectManifest) -> Self {
        LinkedProject::ProjectManifest(v)
    }
}

impl From<ProjectJson> for LinkedProject {
    fn from(v: ProjectJson) -> Self {
        LinkedProject::InlineProjectJson(v)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoverWorkspaceConfig {
    pub command: Vec<String>,
    pub progress_label: String,
    pub files_to_watch: Vec<String>,
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
    pub update_test: bool,
    pub interpret: bool,

    // implementations
    pub implementations: bool,

    // references
    pub method_refs: bool,
    pub refs_adt: bool,   // for Struct, Enum, Union and Trait
    pub refs_trait: bool, // for Struct, Enum, Union and Trait
    pub enum_variant_refs: bool,

    // annotations
    pub location: AnnotationLocation,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationLocation {
    AboveName,
    AboveWholeItem,
}

impl From<AnnotationLocation> for ide::AnnotationLocation {
    fn from(location: AnnotationLocation) -> Self {
        match location {
            AnnotationLocation::AboveName => ide::AnnotationLocation::AboveName,
            AnnotationLocation::AboveWholeItem => ide::AnnotationLocation::AboveWholeItem,
        }
    }
}

impl LensConfig {
    pub fn any(&self) -> bool {
        self.run
            || self.debug
            || self.update_test
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
        self.run || self.debug || self.update_test
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
    pub update_test: bool,
    pub goto_type_def: bool,
}

impl HoverActionsConfig {
    pub const NO_ACTIONS: Self = Self {
        implementations: false,
        references: false,
        run: false,
        debug: false,
        update_test: false,
        goto_type_def: false,
    };

    pub fn any(&self) -> bool {
        self.implementations || self.references || self.runnable() || self.goto_type_def
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn runnable(&self) -> bool {
        self.run || self.debug || self.update_test
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
    /// Additional arguments for the binary being run, if it is a test or benchmark.
    pub extra_test_binary_args: Vec<String>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientCommandsConfig {
    pub run_single: bool,
    pub debug_single: bool,
    pub show_reference: bool,
    pub goto_location: bool,
    pub trigger_parameter_hints: bool,
    pub rename: bool,
}

#[derive(Debug)]
pub enum ConfigErrorInner {
    Json { config_key: String, error: serde_json::Error },
    Toml { config_key: String, error: toml::de::Error },
    ParseError { reason: String },
}

#[derive(Clone, Debug, Default)]
pub struct ConfigErrors(Vec<Arc<ConfigErrorInner>>);

impl ConfigErrors {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for ConfigErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let errors = self.0.iter().format_with("\n", |inner, f| {
            match &**inner {
                ConfigErrorInner::Json { config_key: key, error: e } => {
                    f(key)?;
                    f(&": ")?;
                    f(e)
                }
                ConfigErrorInner::Toml { config_key: key, error: e } => {
                    f(key)?;
                    f(&": ")?;
                    f(e)
                }
                ConfigErrorInner::ParseError { reason } => f(reason),
            }?;
            f(&";")
        });
        write!(f, "invalid config value{}:\n{}", if self.0.len() == 1 { "" } else { "s" }, errors)
    }
}

impl std::error::Error for ConfigErrors {}

impl Config {
    pub fn new(
        root_path: AbsPathBuf,
        caps: lsp_types::ClientCapabilities,
        workspace_roots: Vec<AbsPathBuf>,
        client_info: Option<lsp_types::ClientInfo>,
    ) -> Self {
        static DEFAULT_CONFIG_DATA: OnceLock<&'static DefaultConfigData> = OnceLock::new();

        Config {
            caps: ClientCapabilities::new(caps),
            discovered_projects_from_filesystem: Vec::new(),
            discovered_projects_from_command: Vec::new(),
            root_path,
            snippets: Default::default(),
            workspace_roots,
            client_info: client_info.map(|it| ClientInfo {
                name: it.name,
                version: it.version.as_deref().map(Version::parse).and_then(Result::ok),
            }),
            client_config: (FullConfigInput::default(), ConfigErrors(vec![])),
            default_config: DEFAULT_CONFIG_DATA.get_or_init(|| Box::leak(Box::default())),
            source_root_parent_map: Arc::new(FxHashMap::default()),
            user_config: None,
            detached_files: Default::default(),
            validation_errors: Default::default(),
            ratoml_file: Default::default(),
        }
    }

    pub fn rediscover_workspaces(&mut self) {
        let discovered = ProjectManifest::discover_all(&self.workspace_roots);
        tracing::info!("discovered projects: {:?}", discovered);
        if discovered.is_empty() {
            tracing::error!("failed to find any projects in {:?}", &self.workspace_roots);
        }
        self.discovered_projects_from_filesystem = discovered;
    }

    pub fn remove_workspace(&mut self, path: &AbsPath) {
        if let Some(position) = self.workspace_roots.iter().position(|it| it == path) {
            self.workspace_roots.remove(position);
        }
    }

    pub fn add_workspaces(&mut self, paths: impl Iterator<Item = AbsPathBuf>) {
        self.workspace_roots.extend(paths);
    }

    pub fn json_schema() -> serde_json::Value {
        let mut s = FullConfigInput::json_schema();

        fn sort_objects_by_field(json: &mut serde_json::Value) {
            if let serde_json::Value::Object(object) = json {
                let old = std::mem::take(object);
                old.into_iter().sorted_by(|(k, _), (k2, _)| k.cmp(k2)).for_each(|(k, mut v)| {
                    sort_objects_by_field(&mut v);
                    object.insert(k, v);
                });
            }
        }
        sort_objects_by_field(&mut s);
        s
    }

    pub fn root_path(&self) -> &AbsPathBuf {
        &self.root_path
    }

    pub fn caps(&self) -> &ClientCapabilities {
        &self.caps
    }
}

impl Config {
    pub fn assist(&self, source_root: Option<SourceRootId>) -> AssistConfig {
        AssistConfig {
            snippet_cap: self.snippet_cap(),
            allowed: None,
            insert_use: self.insert_use_config(source_root),
            prefer_no_std: self.imports_preferNoStd(source_root).to_owned(),
            assist_emit_must_use: self.assist_emitMustUse(source_root).to_owned(),
            prefer_prelude: self.imports_preferPrelude(source_root).to_owned(),
            prefer_absolute: self.imports_prefixExternPrelude(source_root).to_owned(),
            term_search_fuel: self.assist_termSearch_fuel(source_root).to_owned() as u64,
            term_search_borrowck: self.assist_termSearch_borrowcheck(source_root).to_owned(),
            code_action_grouping: self.code_action_group(),
            expr_fill_default: match self.assist_expressionFillDefault(source_root) {
                ExprFillDefaultDef::Todo => ExprFillDefaultMode::Todo,
                ExprFillDefaultDef::Default => ExprFillDefaultMode::Default,
                ExprFillDefaultDef::Underscore => ExprFillDefaultMode::Underscore,
            },
        }
    }

    pub fn call_hierarchy(&self) -> CallHierarchyConfig {
        CallHierarchyConfig { exclude_tests: self.references_excludeTests().to_owned() }
    }

    pub fn completion(&self, source_root: Option<SourceRootId>) -> CompletionConfig<'_> {
        let client_capability_fields = self.completion_resolve_support_properties();
        CompletionConfig {
            enable_postfix_completions: self.completion_postfix_enable(source_root).to_owned(),
            enable_imports_on_the_fly: self.completion_autoimport_enable(source_root).to_owned()
                && self.caps.completion_item_edit_resolve(),
            enable_self_on_the_fly: self.completion_autoself_enable(source_root).to_owned(),
            enable_auto_iter: *self.completion_autoIter_enable(source_root),
            enable_auto_await: *self.completion_autoAwait_enable(source_root),
            enable_private_editable: self.completion_privateEditable_enable(source_root).to_owned(),
            full_function_signatures: self
                .completion_fullFunctionSignatures_enable(source_root)
                .to_owned(),
            callable: match self.completion_callable_snippets(source_root) {
                CallableCompletionDef::FillArguments => Some(CallableSnippets::FillArguments),
                CallableCompletionDef::AddParentheses => Some(CallableSnippets::AddParentheses),
                CallableCompletionDef::None => None,
            },
            add_semicolon_to_unit: *self.completion_addSemicolonToUnit(source_root),
            snippet_cap: SnippetCap::new(self.completion_snippet()),
            insert_use: self.insert_use_config(source_root),
            prefer_no_std: self.imports_preferNoStd(source_root).to_owned(),
            prefer_prelude: self.imports_preferPrelude(source_root).to_owned(),
            prefer_absolute: self.imports_prefixExternPrelude(source_root).to_owned(),
            snippets: self.snippets.clone().to_vec(),
            limit: self.completion_limit(source_root).to_owned(),
            enable_term_search: self.completion_termSearch_enable(source_root).to_owned(),
            term_search_fuel: self.completion_termSearch_fuel(source_root).to_owned() as u64,
            fields_to_resolve: if self.client_is_neovim() {
                CompletionFieldsToResolve::empty()
            } else {
                CompletionFieldsToResolve::from_client_capabilities(&client_capability_fields)
            },
            exclude_flyimport: self
                .completion_autoimport_exclude(source_root)
                .iter()
                .map(|it| match it {
                    AutoImportExclusion::Path(path) => {
                        (path.clone(), ide_completion::AutoImportExclusionType::Always)
                    }
                    AutoImportExclusion::Verbose { path, r#type } => (
                        path.clone(),
                        match r#type {
                            AutoImportExclusionType::Always => {
                                ide_completion::AutoImportExclusionType::Always
                            }
                            AutoImportExclusionType::Methods => {
                                ide_completion::AutoImportExclusionType::Methods
                            }
                        },
                    ),
                })
                .collect(),
            exclude_traits: self.completion_excludeTraits(source_root),
        }
    }

    pub fn completion_hide_deprecated(&self) -> bool {
        *self.completion_hideDeprecated(None)
    }

    pub fn detached_files(&self) -> &Vec<AbsPathBuf> {
        // FIXME @alibektas : This is the only config that is confusing. If it's a proper configuration
        // why is it not among the others? If it's client only which I doubt it is current state should be alright
        &self.detached_files
    }

    pub fn diagnostics(&self, source_root: Option<SourceRootId>) -> DiagnosticsConfig {
        DiagnosticsConfig {
            enabled: *self.diagnostics_enable(source_root),
            proc_attr_macros_enabled: self.expand_proc_attr_macros(),
            proc_macros_enabled: *self.procMacro_enable(),
            disable_experimental: !self.diagnostics_experimental_enable(source_root),
            disabled: self.diagnostics_disabled(source_root).clone(),
            expr_fill_default: match self.assist_expressionFillDefault(source_root) {
                ExprFillDefaultDef::Todo => ExprFillDefaultMode::Todo,
                ExprFillDefaultDef::Default => ExprFillDefaultMode::Default,
                ExprFillDefaultDef::Underscore => ExprFillDefaultMode::Underscore,
            },
            snippet_cap: self.snippet_cap(),
            insert_use: self.insert_use_config(source_root),
            prefer_no_std: self.imports_preferNoStd(source_root).to_owned(),
            prefer_prelude: self.imports_preferPrelude(source_root).to_owned(),
            prefer_absolute: self.imports_prefixExternPrelude(source_root).to_owned(),
            style_lints: self.diagnostics_styleLints_enable(source_root).to_owned(),
            term_search_fuel: self.assist_termSearch_fuel(source_root).to_owned() as u64,
            term_search_borrowck: self.assist_termSearch_borrowcheck(source_root).to_owned(),
        }
    }

    pub fn diagnostic_fixes(&self, source_root: Option<SourceRootId>) -> DiagnosticsConfig {
        // We always want to show quickfixes for diagnostics, even when diagnostics/experimental diagnostics are disabled.
        DiagnosticsConfig {
            enabled: true,
            disable_experimental: false,
            ..self.diagnostics(source_root)
        }
    }

    pub fn expand_proc_attr_macros(&self) -> bool {
        self.procMacro_enable().to_owned() && self.procMacro_attributes_enable().to_owned()
    }

    pub fn highlight_related(&self, _source_root: Option<SourceRootId>) -> HighlightRelatedConfig {
        HighlightRelatedConfig {
            references: self.highlightRelated_references_enable().to_owned(),
            break_points: self.highlightRelated_breakPoints_enable().to_owned(),
            exit_points: self.highlightRelated_exitPoints_enable().to_owned(),
            yield_points: self.highlightRelated_yieldPoints_enable().to_owned(),
            closure_captures: self.highlightRelated_closureCaptures_enable().to_owned(),
        }
    }

    pub fn hover_actions(&self) -> HoverActionsConfig {
        let enable = self.caps.hover_actions() && self.hover_actions_enable().to_owned();
        HoverActionsConfig {
            implementations: enable && self.hover_actions_implementations_enable().to_owned(),
            references: enable && self.hover_actions_references_enable().to_owned(),
            run: enable && self.hover_actions_run_enable().to_owned(),
            debug: enable && self.hover_actions_debug_enable().to_owned(),
            update_test: enable
                && self.hover_actions_run_enable().to_owned()
                && self.hover_actions_updateTest_enable().to_owned(),
            goto_type_def: enable && self.hover_actions_gotoTypeDef_enable().to_owned(),
        }
    }

    pub fn hover(&self) -> HoverConfig {
        let mem_kind = |kind| match kind {
            MemoryLayoutHoverRenderKindDef::Both => MemoryLayoutHoverRenderKind::Both,
            MemoryLayoutHoverRenderKindDef::Decimal => MemoryLayoutHoverRenderKind::Decimal,
            MemoryLayoutHoverRenderKindDef::Hexadecimal => MemoryLayoutHoverRenderKind::Hexadecimal,
        };
        HoverConfig {
            links_in_hover: self.hover_links_enable().to_owned(),
            memory_layout: self.hover_memoryLayout_enable().then_some(MemoryLayoutHoverConfig {
                size: self.hover_memoryLayout_size().map(mem_kind),
                offset: self.hover_memoryLayout_offset().map(mem_kind),
                alignment: self.hover_memoryLayout_alignment().map(mem_kind),
                padding: self.hover_memoryLayout_padding().map(mem_kind),
                niches: self.hover_memoryLayout_niches().unwrap_or_default(),
            }),
            documentation: self.hover_documentation_enable().to_owned(),
            format: {
                if self.caps.hover_markdown_support() {
                    HoverDocFormat::Markdown
                } else {
                    HoverDocFormat::PlainText
                }
            },
            keywords: self.hover_documentation_keywords_enable().to_owned(),
            max_trait_assoc_items_count: self.hover_show_traitAssocItems().to_owned(),
            max_fields_count: self.hover_show_fields().to_owned(),
            max_enum_variants_count: self.hover_show_enumVariants().to_owned(),
            max_subst_ty_len: match self.hover_maxSubstitutionLength() {
                Some(MaxSubstitutionLength::Hide) => ide::SubstTyLen::Hide,
                Some(MaxSubstitutionLength::Limit(limit)) => ide::SubstTyLen::LimitTo(*limit),
                None => ide::SubstTyLen::Unlimited,
            },
            show_drop_glue: *self.hover_dropGlue_enable(),
        }
    }

    pub fn inlay_hints(&self) -> InlayHintsConfig {
        let client_capability_fields = self.inlay_hint_resolve_support_properties();

        InlayHintsConfig {
            render_colons: self.inlayHints_renderColons().to_owned(),
            type_hints: self.inlayHints_typeHints_enable().to_owned(),
            sized_bound: self.inlayHints_implicitSizedBoundHints_enable().to_owned(),
            parameter_hints: self.inlayHints_parameterHints_enable().to_owned(),
            generic_parameter_hints: GenericParameterHints {
                type_hints: self.inlayHints_genericParameterHints_type_enable().to_owned(),
                lifetime_hints: self.inlayHints_genericParameterHints_lifetime_enable().to_owned(),
                const_hints: self.inlayHints_genericParameterHints_const_enable().to_owned(),
            },
            chaining_hints: self.inlayHints_chainingHints_enable().to_owned(),
            discriminant_hints: match self.inlayHints_discriminantHints_enable() {
                DiscriminantHintsDef::Always => ide::DiscriminantHints::Always,
                DiscriminantHintsDef::Never => ide::DiscriminantHints::Never,
                DiscriminantHintsDef::Fieldless => ide::DiscriminantHints::Fieldless,
            },
            closure_return_type_hints: match self.inlayHints_closureReturnTypeHints_enable() {
                ClosureReturnTypeHintsDef::Always => ide::ClosureReturnTypeHints::Always,
                ClosureReturnTypeHintsDef::Never => ide::ClosureReturnTypeHints::Never,
                ClosureReturnTypeHintsDef::WithBlock => ide::ClosureReturnTypeHints::WithBlock,
            },
            lifetime_elision_hints: match self.inlayHints_lifetimeElisionHints_enable() {
                LifetimeElisionDef::Always => ide::LifetimeElisionHints::Always,
                LifetimeElisionDef::Never => ide::LifetimeElisionHints::Never,
                LifetimeElisionDef::SkipTrivial => ide::LifetimeElisionHints::SkipTrivial,
            },
            hide_named_constructor_hints: self
                .inlayHints_typeHints_hideNamedConstructor()
                .to_owned(),
            hide_closure_initialization_hints: self
                .inlayHints_typeHints_hideClosureInitialization()
                .to_owned(),
            hide_closure_parameter_hints: self
                .inlayHints_typeHints_hideClosureParameter()
                .to_owned(),
            closure_style: match self.inlayHints_closureStyle() {
                ClosureStyle::ImplFn => hir::ClosureStyle::ImplFn,
                ClosureStyle::RustAnalyzer => hir::ClosureStyle::RANotation,
                ClosureStyle::WithId => hir::ClosureStyle::ClosureWithId,
                ClosureStyle::Hide => hir::ClosureStyle::Hide,
            },
            closure_capture_hints: self.inlayHints_closureCaptureHints_enable().to_owned(),
            adjustment_hints: match self.inlayHints_expressionAdjustmentHints_enable() {
                AdjustmentHintsDef::Always => ide::AdjustmentHints::Always,
                AdjustmentHintsDef::Never => match self.inlayHints_reborrowHints_enable() {
                    ReborrowHintsDef::Always | ReborrowHintsDef::Mutable => {
                        ide::AdjustmentHints::ReborrowOnly
                    }
                    ReborrowHintsDef::Never => ide::AdjustmentHints::Never,
                },
                AdjustmentHintsDef::Reborrow => ide::AdjustmentHints::ReborrowOnly,
            },
            adjustment_hints_mode: match self.inlayHints_expressionAdjustmentHints_mode() {
                AdjustmentHintsModeDef::Prefix => ide::AdjustmentHintsMode::Prefix,
                AdjustmentHintsModeDef::Postfix => ide::AdjustmentHintsMode::Postfix,
                AdjustmentHintsModeDef::PreferPrefix => ide::AdjustmentHintsMode::PreferPrefix,
                AdjustmentHintsModeDef::PreferPostfix => ide::AdjustmentHintsMode::PreferPostfix,
            },
            adjustment_hints_hide_outside_unsafe: self
                .inlayHints_expressionAdjustmentHints_hideOutsideUnsafe()
                .to_owned(),
            binding_mode_hints: self.inlayHints_bindingModeHints_enable().to_owned(),
            param_names_for_lifetime_elision_hints: self
                .inlayHints_lifetimeElisionHints_useParameterNames()
                .to_owned(),
            max_length: self.inlayHints_maxLength().to_owned(),
            closing_brace_hints_min_lines: if self.inlayHints_closingBraceHints_enable().to_owned()
            {
                Some(self.inlayHints_closingBraceHints_minLines().to_owned())
            } else {
                None
            },
            fields_to_resolve: InlayFieldsToResolve::from_client_capabilities(
                &client_capability_fields,
            ),
            implicit_drop_hints: self.inlayHints_implicitDrops_enable().to_owned(),
            range_exclusive_hints: self.inlayHints_rangeExclusiveHints_enable().to_owned(),
        }
    }

    fn insert_use_config(&self, source_root: Option<SourceRootId>) -> InsertUseConfig {
        InsertUseConfig {
            granularity: match self.imports_granularity_group(source_root) {
                ImportGranularityDef::Preserve => ImportGranularity::Preserve,
                ImportGranularityDef::Item => ImportGranularity::Item,
                ImportGranularityDef::Crate => ImportGranularity::Crate,
                ImportGranularityDef::Module => ImportGranularity::Module,
                ImportGranularityDef::One => ImportGranularity::One,
            },
            enforce_granularity: self.imports_granularity_enforce(source_root).to_owned(),
            prefix_kind: match self.imports_prefix(source_root) {
                ImportPrefixDef::Plain => PrefixKind::Plain,
                ImportPrefixDef::ByCrate => PrefixKind::ByCrate,
                ImportPrefixDef::BySelf => PrefixKind::BySelf,
            },
            group: self.imports_group_enable(source_root).to_owned(),
            skip_glob_imports: !self.imports_merge_glob(source_root),
        }
    }

    pub fn join_lines(&self) -> JoinLinesConfig {
        JoinLinesConfig {
            join_else_if: self.joinLines_joinElseIf().to_owned(),
            remove_trailing_comma: self.joinLines_removeTrailingComma().to_owned(),
            unwrap_trivial_blocks: self.joinLines_unwrapTrivialBlock().to_owned(),
            join_assignments: self.joinLines_joinAssignments().to_owned(),
        }
    }

    pub fn highlighting_non_standard_tokens(&self) -> bool {
        self.semanticHighlighting_nonStandardTokens().to_owned()
    }

    pub fn highlighting_config(&self) -> HighlightConfig {
        HighlightConfig {
            strings: self.semanticHighlighting_strings_enable().to_owned(),
            punctuation: self.semanticHighlighting_punctuation_enable().to_owned(),
            specialize_punctuation: self
                .semanticHighlighting_punctuation_specialization_enable()
                .to_owned(),
            macro_bang: self.semanticHighlighting_punctuation_separate_macro_bang().to_owned(),
            operator: self.semanticHighlighting_operator_enable().to_owned(),
            specialize_operator: self
                .semanticHighlighting_operator_specialization_enable()
                .to_owned(),
            inject_doc_comment: self.semanticHighlighting_doc_comment_inject_enable().to_owned(),
            syntactic_name_ref_highlighting: false,
        }
    }

    pub fn has_linked_projects(&self) -> bool {
        !self.linkedProjects().is_empty()
    }

    pub fn linked_manifests(&self) -> impl Iterator<Item = &Utf8Path> + '_ {
        self.linkedProjects().iter().filter_map(|it| match it {
            ManifestOrProjectJson::Manifest(p) => Some(&**p),
            // despite having a buildfile, using this variant as a manifest
            // will fail.
            ManifestOrProjectJson::DiscoveredProjectJson { .. } => None,
            ManifestOrProjectJson::ProjectJson { .. } => None,
        })
    }

    pub fn has_linked_project_jsons(&self) -> bool {
        self.linkedProjects()
            .iter()
            .any(|it| matches!(it, ManifestOrProjectJson::ProjectJson { .. }))
    }

    pub fn discover_workspace_config(&self) -> Option<&DiscoverWorkspaceConfig> {
        self.workspace_discoverConfig().as_ref()
    }

    fn discovered_projects(&self) -> Vec<ManifestOrProjectJson> {
        let exclude_dirs: Vec<_> =
            self.files_exclude().iter().map(|p| self.root_path.join(p)).collect();

        let mut projects = vec![];
        for fs_proj in &self.discovered_projects_from_filesystem {
            let manifest_path = fs_proj.manifest_path();
            if exclude_dirs.iter().any(|p| manifest_path.starts_with(p)) {
                continue;
            }

            let buf: Utf8PathBuf = manifest_path.to_path_buf().into();
            projects.push(ManifestOrProjectJson::Manifest(buf));
        }

        for dis_proj in &self.discovered_projects_from_command {
            projects.push(ManifestOrProjectJson::DiscoveredProjectJson {
                data: dis_proj.data.clone(),
                buildfile: dis_proj.buildfile.clone(),
            });
        }

        projects
    }

    pub fn linked_or_discovered_projects(&self) -> Vec<LinkedProject> {
        let linked_projects = self.linkedProjects();
        let projects = if linked_projects.is_empty() {
            self.discovered_projects()
        } else {
            linked_projects.clone()
        };

        projects
            .iter()
            .filter_map(|linked_project| match linked_project {
                ManifestOrProjectJson::Manifest(it) => {
                    let path = self.root_path.join(it);
                    ProjectManifest::from_manifest_file(path)
                        .map_err(|e| tracing::error!("failed to load linked project: {}", e))
                        .ok()
                        .map(Into::into)
                }
                ManifestOrProjectJson::DiscoveredProjectJson { data, buildfile } => {
                    let root_path = buildfile.parent().expect("Unable to get parent of buildfile");

                    Some(ProjectJson::new(None, root_path, data.clone()).into())
                }
                ManifestOrProjectJson::ProjectJson(it) => {
                    Some(ProjectJson::new(None, &self.root_path, it.clone()).into())
                }
            })
            .collect()
    }

    pub fn prefill_caches(&self) -> bool {
        self.cachePriming_enable().to_owned()
    }

    pub fn publish_diagnostics(&self, source_root: Option<SourceRootId>) -> bool {
        self.diagnostics_enable(source_root).to_owned()
    }

    pub fn diagnostics_map(&self, source_root: Option<SourceRootId>) -> DiagnosticsMapConfig {
        DiagnosticsMapConfig {
            remap_prefix: self.diagnostics_remapPrefix(source_root).clone(),
            warnings_as_info: self.diagnostics_warningsAsInfo(source_root).clone(),
            warnings_as_hint: self.diagnostics_warningsAsHint(source_root).clone(),
            check_ignore: self.check_ignore(source_root).clone(),
        }
    }

    pub fn extra_args(&self, source_root: Option<SourceRootId>) -> &Vec<String> {
        self.cargo_extraArgs(source_root)
    }

    pub fn extra_env(
        &self,
        source_root: Option<SourceRootId>,
    ) -> &FxHashMap<String, Option<String>> {
        self.cargo_extraEnv(source_root)
    }

    pub fn check_extra_args(&self, source_root: Option<SourceRootId>) -> Vec<String> {
        let mut extra_args = self.extra_args(source_root).clone();
        extra_args.extend_from_slice(self.check_extraArgs(source_root));
        extra_args
    }

    pub fn check_extra_env(
        &self,
        source_root: Option<SourceRootId>,
    ) -> FxHashMap<String, Option<String>> {
        let mut extra_env = self.cargo_extraEnv(source_root).clone();
        extra_env.extend(self.check_extraEnv(source_root).clone());
        extra_env
    }

    pub fn lru_parse_query_capacity(&self) -> Option<u16> {
        self.lru_capacity().to_owned()
    }

    pub fn lru_query_capacities_config(&self) -> Option<&FxHashMap<Box<str>, u16>> {
        self.lru_query_capacities().is_empty().not().then(|| self.lru_query_capacities())
    }

    pub fn proc_macro_srv(&self) -> Option<AbsPathBuf> {
        let path = self.procMacro_server().clone()?;
        Some(AbsPathBuf::try_from(path).unwrap_or_else(|path| self.root_path.join(path)))
    }

    pub fn ignored_proc_macros(
        &self,
        source_root: Option<SourceRootId>,
    ) -> &FxHashMap<Box<str>, Box<[Box<str>]>> {
        self.procMacro_ignored(source_root)
    }

    pub fn expand_proc_macros(&self) -> bool {
        self.procMacro_enable().to_owned()
    }

    pub fn files(&self) -> FilesConfig {
        FilesConfig {
            watcher: match self.files_watcher() {
                FilesWatcherDef::Client if self.did_change_watched_files_dynamic_registration() => {
                    FilesWatcher::Client
                }
                _ => FilesWatcher::Server,
            },
            exclude: self.excluded().collect(),
        }
    }

    pub fn excluded(&self) -> impl Iterator<Item = AbsPathBuf> + use<'_> {
        self.files_exclude().iter().map(|it| self.root_path.join(it))
    }

    pub fn notifications(&self) -> NotificationsConfig {
        NotificationsConfig {
            cargo_toml_not_found: self.notifications_cargoTomlNotFound().to_owned(),
        }
    }

    pub fn cargo_autoreload_config(&self, source_root: Option<SourceRootId>) -> bool {
        self.cargo_autoreload(source_root).to_owned()
    }

    pub fn run_build_scripts(&self, source_root: Option<SourceRootId>) -> bool {
        self.cargo_buildScripts_enable(source_root).to_owned() || self.procMacro_enable().to_owned()
    }

    pub fn cargo(&self, source_root: Option<SourceRootId>) -> CargoConfig {
        let rustc_source = self.rustc_source(source_root).as_ref().map(|rustc_src| {
            if rustc_src == "discover" {
                RustLibSource::Discover
            } else {
                RustLibSource::Path(self.root_path.join(rustc_src))
            }
        });
        let sysroot = self.cargo_sysroot(source_root).as_ref().map(|sysroot| {
            if sysroot == "discover" {
                RustLibSource::Discover
            } else {
                RustLibSource::Path(self.root_path.join(sysroot))
            }
        });
        let sysroot_src =
            self.cargo_sysrootSrc(source_root).as_ref().map(|sysroot| self.root_path.join(sysroot));
        let extra_includes = self
            .vfs_extraIncludes(source_root)
            .iter()
            .map(String::as_str)
            .map(AbsPathBuf::try_from)
            .filter_map(Result::ok)
            .collect();

        CargoConfig {
            all_targets: *self.cargo_allTargets(source_root),
            features: match &self.cargo_features(source_root) {
                CargoFeaturesDef::All => CargoFeatures::All,
                CargoFeaturesDef::Selected(features) => CargoFeatures::Selected {
                    features: features.clone(),
                    no_default_features: self.cargo_noDefaultFeatures(source_root).to_owned(),
                },
            },
            target: self.cargo_target(source_root).clone(),
            sysroot,
            sysroot_src,
            rustc_source,
            extra_includes,
            cfg_overrides: project_model::CfgOverrides {
                global: {
                    let (enabled, disabled): (Vec<_>, Vec<_>) =
                        self.cargo_cfgs(source_root).iter().partition_map(|s| {
                            s.strip_prefix("!").map_or(Either::Left(s), Either::Right)
                        });
                    CfgDiff::new(
                        enabled
                            .into_iter()
                            // parse any cfg setting formatted as key=value or just key (without value)
                            .map(|s| match s.split_once("=") {
                                Some((key, val)) => CfgAtom::KeyValue {
                                    key: Symbol::intern(key),
                                    value: Symbol::intern(val),
                                },
                                None => CfgAtom::Flag(Symbol::intern(s)),
                            })
                            .collect(),
                        disabled
                            .into_iter()
                            .map(|s| match s.split_once("=") {
                                Some((key, val)) => CfgAtom::KeyValue {
                                    key: Symbol::intern(key),
                                    value: Symbol::intern(val),
                                },
                                None => CfgAtom::Flag(Symbol::intern(s)),
                            })
                            .collect(),
                    )
                },
                selective: Default::default(),
            },
            wrap_rustc_in_build_scripts: *self.cargo_buildScripts_useRustcWrapper(source_root),
            invocation_strategy: match self.cargo_buildScripts_invocationStrategy(source_root) {
                InvocationStrategy::Once => project_model::InvocationStrategy::Once,
                InvocationStrategy::PerWorkspace => project_model::InvocationStrategy::PerWorkspace,
            },
            run_build_script_command: self.cargo_buildScripts_overrideCommand(source_root).clone(),
            extra_args: self.cargo_extraArgs(source_root).clone(),
            extra_env: self.cargo_extraEnv(source_root).clone(),
            target_dir: self.target_dir_from_config(source_root),
            set_test: *self.cfg_setTest(source_root),
            no_deps: *self.cargo_noDeps(source_root),
        }
    }

    pub fn cfg_set_test(&self, source_root: Option<SourceRootId>) -> bool {
        *self.cfg_setTest(source_root)
    }

    pub(crate) fn completion_snippets_default() -> FxIndexMap<String, SnippetDef> {
        serde_json::from_str(
            r#"{
            "Ok": {
                "postfix": "ok",
                "body": "Ok(${receiver})",
                "description": "Wrap the expression in a `Result::Ok`",
                "scope": "expr"
            },
            "Box::pin": {
                "postfix": "pinbox",
                "body": "Box::pin(${receiver})",
                "requires": "std::boxed::Box",
                "description": "Put the expression into a pinned `Box`",
                "scope": "expr"
            },
            "Arc::new": {
                "postfix": "arc",
                "body": "Arc::new(${receiver})",
                "requires": "std::sync::Arc",
                "description": "Put the expression into an `Arc`",
                "scope": "expr"
            },
            "Some": {
                "postfix": "some",
                "body": "Some(${receiver})",
                "description": "Wrap the expression in an `Option::Some`",
                "scope": "expr"
            },
            "Err": {
                "postfix": "err",
                "body": "Err(${receiver})",
                "description": "Wrap the expression in a `Result::Err`",
                "scope": "expr"
            },
            "Rc::new": {
                "postfix": "rc",
                "body": "Rc::new(${receiver})",
                "requires": "std::rc::Rc",
                "description": "Put the expression into an `Rc`",
                "scope": "expr"
            }
        }"#,
        )
        .unwrap()
    }

    pub fn rustfmt(&self, source_root_id: Option<SourceRootId>) -> RustfmtConfig {
        match &self.rustfmt_overrideCommand(source_root_id) {
            Some(args) if !args.is_empty() => {
                let mut args = args.clone();
                let command = args.remove(0);
                RustfmtConfig::CustomCommand { command, args }
            }
            Some(_) | None => RustfmtConfig::Rustfmt {
                extra_args: self.rustfmt_extraArgs(source_root_id).clone(),
                enable_range_formatting: *self.rustfmt_rangeFormatting_enable(source_root_id),
            },
        }
    }

    pub fn flycheck_workspace(&self, source_root: Option<SourceRootId>) -> bool {
        *self.check_workspace(source_root)
    }

    pub(crate) fn cargo_test_options(&self, source_root: Option<SourceRootId>) -> CargoOptions {
        CargoOptions {
            target_tuples: self.cargo_target(source_root).clone().into_iter().collect(),
            all_targets: false,
            no_default_features: *self.cargo_noDefaultFeatures(source_root),
            all_features: matches!(self.cargo_features(source_root), CargoFeaturesDef::All),
            features: match self.cargo_features(source_root).clone() {
                CargoFeaturesDef::All => vec![],
                CargoFeaturesDef::Selected(it) => it,
            },
            extra_args: self.extra_args(source_root).clone(),
            extra_test_bin_args: self.runnables_extraTestBinaryArgs(source_root).clone(),
            extra_env: self.extra_env(source_root).clone(),
            target_dir: self.target_dir_from_config(source_root),
        }
    }

    pub(crate) fn flycheck(&self, source_root: Option<SourceRootId>) -> FlycheckConfig {
        match &self.check_overrideCommand(source_root) {
            Some(args) if !args.is_empty() => {
                let mut args = args.clone();
                let command = args.remove(0);
                FlycheckConfig::CustomCommand {
                    command,
                    args,
                    extra_env: self.check_extra_env(source_root),
                    invocation_strategy: match self.check_invocationStrategy(source_root) {
                        InvocationStrategy::Once => crate::flycheck::InvocationStrategy::Once,
                        InvocationStrategy::PerWorkspace => {
                            crate::flycheck::InvocationStrategy::PerWorkspace
                        }
                    },
                }
            }
            Some(_) | None => FlycheckConfig::CargoCommand {
                command: self.check_command(source_root).clone(),
                options: CargoOptions {
                    target_tuples: self
                        .check_targets(source_root)
                        .clone()
                        .and_then(|targets| match &targets.0[..] {
                            [] => None,
                            targets => Some(targets.into()),
                        })
                        .unwrap_or_else(|| {
                            self.cargo_target(source_root).clone().into_iter().collect()
                        }),
                    all_targets: self
                        .check_allTargets(source_root)
                        .unwrap_or(*self.cargo_allTargets(source_root)),
                    no_default_features: self
                        .check_noDefaultFeatures(source_root)
                        .unwrap_or(*self.cargo_noDefaultFeatures(source_root)),
                    all_features: matches!(
                        self.check_features(source_root)
                            .as_ref()
                            .unwrap_or(self.cargo_features(source_root)),
                        CargoFeaturesDef::All
                    ),
                    features: match self
                        .check_features(source_root)
                        .clone()
                        .unwrap_or_else(|| self.cargo_features(source_root).clone())
                    {
                        CargoFeaturesDef::All => vec![],
                        CargoFeaturesDef::Selected(it) => it,
                    },
                    extra_args: self.check_extra_args(source_root),
                    extra_test_bin_args: self.runnables_extraTestBinaryArgs(source_root).clone(),
                    extra_env: self.check_extra_env(source_root),
                    target_dir: self.target_dir_from_config(source_root),
                },
                ansi_color_output: self.color_diagnostic_output(),
            },
        }
    }

    fn target_dir_from_config(&self, source_root: Option<SourceRootId>) -> Option<Utf8PathBuf> {
        self.cargo_targetDir(source_root).as_ref().and_then(|target_dir| match target_dir {
            TargetDirectory::UseSubdirectory(true) => {
                let env_var = env::var("CARGO_TARGET_DIR").ok();
                let mut path = Utf8PathBuf::from(env_var.as_deref().unwrap_or("target"));
                path.push("rust-analyzer");
                Some(path)
            }
            TargetDirectory::UseSubdirectory(false) => None,
            TargetDirectory::Directory(dir) => Some(dir.clone()),
        })
    }

    pub fn check_on_save(&self, source_root: Option<SourceRootId>) -> bool {
        *self.checkOnSave(source_root)
    }

    pub fn script_rebuild_on_save(&self, source_root: Option<SourceRootId>) -> bool {
        *self.cargo_buildScripts_rebuildOnSave(source_root)
    }

    pub fn runnables(&self, source_root: Option<SourceRootId>) -> RunnablesConfig {
        RunnablesConfig {
            override_cargo: self.runnables_command(source_root).clone(),
            cargo_extra_args: self.runnables_extraArgs(source_root).clone(),
            extra_test_binary_args: self.runnables_extraTestBinaryArgs(source_root).clone(),
        }
    }

    pub fn find_all_refs_exclude_imports(&self) -> bool {
        *self.references_excludeImports()
    }

    pub fn find_all_refs_exclude_tests(&self) -> bool {
        *self.references_excludeTests()
    }

    pub fn snippet_cap(&self) -> Option<SnippetCap> {
        // FIXME: Also detect the proposed lsp version at caps.workspace.workspaceEdit.snippetEditSupport
        // once lsp-types has it.
        SnippetCap::new(self.snippet_text_edit())
    }

    pub fn call_info(&self) -> CallInfoConfig {
        CallInfoConfig {
            params_only: matches!(self.signatureInfo_detail(), SignatureDetail::Parameters),
            docs: *self.signatureInfo_documentation_enable(),
        }
    }

    pub fn lens(&self) -> LensConfig {
        LensConfig {
            run: *self.lens_enable() && *self.lens_run_enable(),
            debug: *self.lens_enable() && *self.lens_debug_enable(),
            update_test: *self.lens_enable()
                && *self.lens_updateTest_enable()
                && *self.lens_run_enable(),
            interpret: *self.lens_enable() && *self.lens_run_enable() && *self.interpret_tests(),
            implementations: *self.lens_enable() && *self.lens_implementations_enable(),
            method_refs: *self.lens_enable() && *self.lens_references_method_enable(),
            refs_adt: *self.lens_enable() && *self.lens_references_adt_enable(),
            refs_trait: *self.lens_enable() && *self.lens_references_trait_enable(),
            enum_variant_refs: *self.lens_enable() && *self.lens_references_enumVariant_enable(),
            location: *self.lens_location(),
        }
    }

    pub fn workspace_symbol(&self, source_root: Option<SourceRootId>) -> WorkspaceSymbolConfig {
        WorkspaceSymbolConfig {
            search_scope: match self.workspace_symbol_search_scope(source_root) {
                WorkspaceSymbolSearchScopeDef::Workspace => WorkspaceSymbolSearchScope::Workspace,
                WorkspaceSymbolSearchScopeDef::WorkspaceAndDependencies => {
                    WorkspaceSymbolSearchScope::WorkspaceAndDependencies
                }
            },
            search_kind: match self.workspace_symbol_search_kind(source_root) {
                WorkspaceSymbolSearchKindDef::OnlyTypes => WorkspaceSymbolSearchKind::OnlyTypes,
                WorkspaceSymbolSearchKindDef::AllSymbols => WorkspaceSymbolSearchKind::AllSymbols,
            },
            search_limit: *self.workspace_symbol_search_limit(source_root),
        }
    }

    pub fn client_commands(&self) -> ClientCommandsConfig {
        let commands = self.commands().map(|it| it.commands).unwrap_or_default();

        let get = |name: &str| commands.iter().any(|it| it == name);

        ClientCommandsConfig {
            run_single: get("rust-analyzer.runSingle"),
            debug_single: get("rust-analyzer.debugSingle"),
            show_reference: get("rust-analyzer.showReferences"),
            goto_location: get("rust-analyzer.gotoLocation"),
            trigger_parameter_hints: get("rust-analyzer.triggerParameterHints"),
            rename: get("rust-analyzer.rename"),
        }
    }

    pub fn prime_caches_num_threads(&self) -> usize {
        match self.cachePriming_numThreads() {
            NumThreads::Concrete(0) | NumThreads::Physical => num_cpus::get_physical(),
            &NumThreads::Concrete(n) => n,
            NumThreads::Logical => num_cpus::get(),
        }
    }

    pub fn main_loop_num_threads(&self) -> usize {
        match self.numThreads() {
            Some(NumThreads::Concrete(0)) | None | Some(NumThreads::Physical) => {
                num_cpus::get_physical()
            }
            &Some(NumThreads::Concrete(n)) => n,
            Some(NumThreads::Logical) => num_cpus::get(),
        }
    }

    pub fn typing_trigger_chars(&self) -> &str {
        self.typing_triggerChars().as_deref().unwrap_or_default()
    }

    // VSCode is our reference implementation, so we allow ourselves to work around issues by
    // special casing certain versions
    pub fn visual_studio_code_version(&self) -> Option<&Version> {
        self.client_info
            .as_ref()
            .filter(|it| it.name.starts_with("Visual Studio Code"))
            .and_then(|it| it.version.as_ref())
    }

    pub fn client_is_helix(&self) -> bool {
        self.client_info.as_ref().map(|it| it.name == "helix").unwrap_or_default()
    }

    pub fn client_is_neovim(&self) -> bool {
        self.client_info.as_ref().map(|it| it.name == "Neovim").unwrap_or_default()
    }
}
// Deserialization definitions

macro_rules! create_bool_or_string_serde {
    ($ident:ident<$bool:literal, $string:literal>) => {
        mod $ident {
            pub(super) fn deserialize<'de, D>(d: D) -> Result<(), D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct V;
                impl<'de> serde::de::Visitor<'de> for V {
                    type Value = ();

                    fn expecting(
                        &self,
                        formatter: &mut std::fmt::Formatter<'_>,
                    ) -> std::fmt::Result {
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

            pub(super) fn serialize<S>(serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str($string)
            }
        }
    };
}
create_bool_or_string_serde!(true_or_always<true, "always">);
create_bool_or_string_serde!(false_or_never<false, "never">);

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
enum SnippetScopeDef {
    #[default]
    Expr,
    Item,
    Type,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub(crate) struct SnippetDef {
    #[serde(with = "single_or_array")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    prefix: Vec<String>,

    #[serde(with = "single_or_array")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    postfix: Vec<String>,

    #[serde(with = "single_or_array")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    body: Vec<String>,

    #[serde(with = "single_or_array")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    requires: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,

    scope: SnippetScopeDef,
}

mod single_or_array {
    use serde::{Deserialize, Serialize};

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
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

    pub(super) fn serialize<S>(vec: &[String], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match vec {
            // []  case is handled by skip_serializing_if
            [single] => serializer.serialize_str(single),
            slice => slice.serialize(serializer),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
enum ManifestOrProjectJson {
    Manifest(Utf8PathBuf),
    ProjectJson(ProjectJsonData),
    DiscoveredProjectJson {
        data: ProjectJsonData,
        #[serde(serialize_with = "serialize_abs_pathbuf")]
        #[serde(deserialize_with = "deserialize_abs_pathbuf")]
        buildfile: AbsPathBuf,
    },
}

fn deserialize_abs_pathbuf<'de, D>(de: D) -> std::result::Result<AbsPathBuf, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    let path = String::deserialize(de)?;

    AbsPathBuf::try_from(path.as_ref())
        .map_err(|err| serde::de::Error::custom(format!("invalid path name: {err:?}")))
}

fn serialize_abs_pathbuf<S>(path: &AbsPathBuf, se: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let path: &Utf8Path = path.as_ref();
    se.serialize_str(path.as_str())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ExprFillDefaultDef {
    Todo,
    Default,
    Underscore,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
pub enum AutoImportExclusion {
    Path(String),
    Verbose { path: String, r#type: AutoImportExclusionType },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum AutoImportExclusionType {
    Always,
    Methods,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ImportGranularityDef {
    Preserve,
    Item,
    Crate,
    Module,
    One,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CallableCompletionDef {
    FillArguments,
    AddParentheses,
    None,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum CargoFeaturesDef {
    All,
    #[serde(untagged)]
    Selected(Vec<String>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InvocationStrategy {
    Once,
    PerWorkspace,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CheckOnSaveTargets(#[serde(with = "single_or_array")] Vec<String>);

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum LifetimeElisionDef {
    SkipTrivial,
    #[serde(with = "true_or_always")]
    #[serde(untagged)]
    Always,
    #[serde(with = "false_or_never")]
    #[serde(untagged)]
    Never,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ClosureReturnTypeHintsDef {
    WithBlock,
    #[serde(with = "true_or_always")]
    #[serde(untagged)]
    Always,
    #[serde(with = "false_or_never")]
    #[serde(untagged)]
    Never,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ClosureStyle {
    ImplFn,
    RustAnalyzer,
    WithId,
    Hide,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ReborrowHintsDef {
    Mutable,
    #[serde(with = "true_or_always")]
    #[serde(untagged)]
    Always,
    #[serde(with = "false_or_never")]
    #[serde(untagged)]
    Never,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum AdjustmentHintsDef {
    Reborrow,
    #[serde(with = "true_or_always")]
    #[serde(untagged)]
    Always,
    #[serde(with = "false_or_never")]
    #[serde(untagged)]
    Never,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum DiscriminantHintsDef {
    Fieldless,
    #[serde(with = "true_or_always")]
    #[serde(untagged)]
    Always,
    #[serde(with = "false_or_never")]
    #[serde(untagged)]
    Never,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum AdjustmentHintsModeDef {
    Prefix,
    Postfix,
    PreferPrefix,
    PreferPostfix,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum FilesWatcherDef {
    Client,
    Notify,
    Server,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum ImportPrefixDef {
    Plain,
    #[serde(rename = "self")]
    #[serde(alias = "by_self")]
    BySelf,
    #[serde(rename = "crate")]
    #[serde(alias = "by_crate")]
    ByCrate,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum WorkspaceSymbolSearchScopeDef {
    Workspace,
    WorkspaceAndDependencies,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum SignatureDetail {
    Full,
    Parameters,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
enum WorkspaceSymbolSearchKindDef {
    OnlyTypes,
    AllSymbols,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
enum MemoryLayoutHoverRenderKindDef {
    Decimal,
    Hexadecimal,
    Both,
}

#[test]
fn untagged_option_hover_render_kind() {
    let hex = MemoryLayoutHoverRenderKindDef::Hexadecimal;

    let ser = serde_json::to_string(&Some(hex)).unwrap();
    assert_eq!(&ser, "\"hexadecimal\"");

    let opt: Option<_> = serde_json::from_str("\"hexadecimal\"").unwrap();
    assert_eq!(opt, Some(hex));
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(untagged)]
pub enum TargetDirectory {
    UseSubdirectory(bool),
    Directory(Utf8PathBuf),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NumThreads {
    Physical,
    Logical,
    #[serde(untagged)]
    Concrete(usize),
}

macro_rules! _default_val {
    ($default:expr, $ty:ty) => {{
        let default_: $ty = $default;
        default_
    }};
}
use _default_val as default_val;

macro_rules! _default_str {
    ($default:expr, $ty:ty) => {{
        let val = default_val!($default, $ty);
        serde_json::to_string_pretty(&val).unwrap()
    }};
}
use _default_str as default_str;

macro_rules! _impl_for_config_data {
    (local, $(
            $(#[doc=$doc:literal])*
            $vis:vis $field:ident : $ty:ty = $default:expr,
        )*
    ) => {
        impl Config {
            $(
                $($doc)*
                #[allow(non_snake_case)]
                $vis fn $field(&self, source_root: Option<SourceRootId>) -> &$ty {
                    let mut source_root = source_root.as_ref();
                    while let Some(sr) = source_root {
                        if let Some((file, _)) = self.ratoml_file.get(&sr) {
                            match file {
                                RatomlFile::Workspace(config) => {
                                    if let Some(v) = config.local.$field.as_ref() {
                                        return &v;
                                    }
                                },
                                RatomlFile::Crate(config) => {
                                    if let Some(value) = config.$field.as_ref() {
                                        return value;
                                    }
                                }
                            }
                        }
                        source_root = self.source_root_parent_map.get(&sr);
                    }

                    if let Some(v) = self.client_config.0.local.$field.as_ref() {
                        return &v;
                    }

                    if let Some((user_config, _)) = self.user_config.as_ref() {
                        if let Some(v) = user_config.local.$field.as_ref() {
                            return &v;
                        }
                    }

                    &self.default_config.local.$field
                }
            )*
        }
    };
    (workspace, $(
            $(#[doc=$doc:literal])*
            $vis:vis $field:ident : $ty:ty = $default:expr,
        )*
    ) => {
        impl Config {
            $(
                $($doc)*
                #[allow(non_snake_case)]
                $vis fn $field(&self, source_root: Option<SourceRootId>) -> &$ty {
                    let mut source_root = source_root.as_ref();
                    while let Some(sr) = source_root {
                        if let Some((RatomlFile::Workspace(config), _)) = self.ratoml_file.get(&sr) {
                            if let Some(v) = config.workspace.$field.as_ref() {
                                return &v;
                            }
                        }
                        source_root = self.source_root_parent_map.get(&sr);
                    }

                    if let Some(v) = self.client_config.0.workspace.$field.as_ref() {
                        return &v;
                    }

                    if let Some((user_config, _)) = self.user_config.as_ref() {
                        if let Some(v) = user_config.workspace.$field.as_ref() {
                            return &v;
                        }
                    }

                    &self.default_config.workspace.$field
                }
            )*
        }
    };
    (global, $(
            $(#[doc=$doc:literal])*
            $vis:vis $field:ident : $ty:ty = $default:expr,
        )*
    ) => {
        impl Config {
            $(
                $($doc)*
                #[allow(non_snake_case)]
                $vis fn $field(&self) -> &$ty {
                    if let Some(v) = self.client_config.0.global.$field.as_ref() {
                        return &v;
                    }

                    if let Some((user_config, _)) = self.user_config.as_ref() {
                        if let Some(v) = user_config.global.$field.as_ref() {
                            return &v;
                        }
                    }


                    &self.default_config.global.$field
                }
            )*
        }
    };
    (client, $(
            $(#[doc=$doc:literal])*
            $vis:vis $field:ident : $ty:ty = $default:expr,
       )*
    ) => {
        impl Config {
            $(
                $($doc)*
                #[allow(non_snake_case)]
                $vis fn $field(&self) -> &$ty {
                    if let Some(v) = self.client_config.0.client.$field.as_ref() {
                        return &v;
                    }

                    &self.default_config.client.$field
                }
            )*
        }
    };
}
use _impl_for_config_data as impl_for_config_data;

macro_rules! _config_data {
    // modname is for the tests
    ($(#[doc=$dox:literal])* $modname:ident: struct $name:ident <- $input:ident -> {
        $(
            $(#[doc=$doc:literal])*
            $vis:vis $field:ident $(| $alias:ident)*: $ty:ty = $default:expr,
        )*
    }) => {
        /// Default config values for this grouping.
        #[allow(non_snake_case)]
        #[derive(Debug, Clone )]
        struct $name { $($field: $ty,)* }

        impl_for_config_data!{
            $modname,
            $(
                $vis $field : $ty = $default,
            )*
        }

        /// All fields `Option<T>`, `None` representing fields not set in a particular JSON/TOML blob.
        #[allow(non_snake_case)]
        #[derive(Clone, Default)]
        struct $input { $(
            $field: Option<$ty>,
        )* }

        impl std::fmt::Debug for $input {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut s = f.debug_struct(stringify!($input));
                $(
                    if let Some(val) = self.$field.as_ref() {
                        s.field(stringify!($field), val);
                    }
                )*
                s.finish()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                $name {$(
                    $field: default_val!($default, $ty),
                )*}
            }
        }

        #[allow(unused, clippy::ptr_arg)]
        impl $input {
            const FIELDS: &'static [&'static str] = &[$(stringify!($field)),*];

            fn from_json(json: &mut serde_json::Value, error_sink: &mut Vec<(String, serde_json::Error)>) -> Self {
                Self {$(
                    $field: get_field_json(
                        json,
                        error_sink,
                        stringify!($field),
                        None$(.or(Some(stringify!($alias))))*,
                    ),
                )*}
            }

            fn from_toml(toml: &toml::Table, error_sink: &mut Vec<(String, toml::de::Error)>) -> Self {
                Self {$(
                    $field: get_field_toml::<$ty>(
                        toml,
                        error_sink,
                        stringify!($field),
                        None$(.or(Some(stringify!($alias))))*,
                    ),
                )*}
            }

            fn schema_fields(sink: &mut Vec<SchemaField>) {
                sink.extend_from_slice(&[
                    $({
                        let field = stringify!($field);
                        let ty = stringify!($ty);
                        let default = default_str!($default, $ty);

                        (field, ty, &[$($doc),*], default)
                    },)*
                ])
            }
        }

        mod $modname {
            #[test]
            fn fields_are_sorted() {
                super::$input::FIELDS.windows(2).for_each(|w| assert!(w[0] <= w[1], "{} <= {} does not hold", w[0], w[1]));
            }
        }
    };
}
use _config_data as config_data;

#[derive(Default, Debug, Clone)]
struct DefaultConfigData {
    global: GlobalDefaultConfigData,
    workspace: WorkspaceDefaultConfigData,
    local: LocalDefaultConfigData,
    client: ClientDefaultConfigData,
}

/// All of the config levels, all fields `Option<T>`, to describe fields that are actually set by
/// some rust-analyzer.toml file or JSON blob. An empty rust-analyzer.toml corresponds to
/// all fields being None.
#[derive(Debug, Clone, Default)]
struct FullConfigInput {
    global: GlobalConfigInput,
    workspace: WorkspaceConfigInput,
    local: LocalConfigInput,
    client: ClientConfigInput,
}

impl FullConfigInput {
    fn from_json(
        mut json: serde_json::Value,
        error_sink: &mut Vec<(String, serde_json::Error)>,
    ) -> FullConfigInput {
        FullConfigInput {
            global: GlobalConfigInput::from_json(&mut json, error_sink),
            local: LocalConfigInput::from_json(&mut json, error_sink),
            client: ClientConfigInput::from_json(&mut json, error_sink),
            workspace: WorkspaceConfigInput::from_json(&mut json, error_sink),
        }
    }

    fn schema_fields() -> Vec<SchemaField> {
        let mut fields = Vec::new();
        GlobalConfigInput::schema_fields(&mut fields);
        LocalConfigInput::schema_fields(&mut fields);
        ClientConfigInput::schema_fields(&mut fields);
        WorkspaceConfigInput::schema_fields(&mut fields);
        fields.sort_by_key(|&(x, ..)| x);
        fields
            .iter()
            .tuple_windows()
            .for_each(|(a, b)| assert!(a.0 != b.0, "{a:?} duplicate field"));
        fields
    }

    fn json_schema() -> serde_json::Value {
        schema(&Self::schema_fields())
    }

    #[cfg(test)]
    fn manual() -> String {
        manual(&Self::schema_fields())
    }
}

/// All of the config levels, all fields `Option<T>`, to describe fields that are actually set by
/// some rust-analyzer.toml file or JSON blob. An empty rust-analyzer.toml corresponds to
/// all fields being None.
#[derive(Debug, Clone, Default)]
struct GlobalWorkspaceLocalConfigInput {
    global: GlobalConfigInput,
    local: LocalConfigInput,
    workspace: WorkspaceConfigInput,
}

impl GlobalWorkspaceLocalConfigInput {
    const FIELDS: &'static [&'static [&'static str]] =
        &[GlobalConfigInput::FIELDS, LocalConfigInput::FIELDS];
    fn from_toml(
        toml: toml::Table,
        error_sink: &mut Vec<(String, toml::de::Error)>,
    ) -> GlobalWorkspaceLocalConfigInput {
        GlobalWorkspaceLocalConfigInput {
            global: GlobalConfigInput::from_toml(&toml, error_sink),
            local: LocalConfigInput::from_toml(&toml, error_sink),
            workspace: WorkspaceConfigInput::from_toml(&toml, error_sink),
        }
    }
}

/// Workspace and local config levels, all fields `Option<T>`, to describe fields that are actually set by
/// some rust-analyzer.toml file or JSON blob. An empty rust-analyzer.toml corresponds to
/// all fields being None.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct WorkspaceLocalConfigInput {
    workspace: WorkspaceConfigInput,
    local: LocalConfigInput,
}

impl WorkspaceLocalConfigInput {
    #[allow(dead_code)]
    const FIELDS: &'static [&'static [&'static str]] =
        &[WorkspaceConfigInput::FIELDS, LocalConfigInput::FIELDS];
    fn from_toml(toml: toml::Table, error_sink: &mut Vec<(String, toml::de::Error)>) -> Self {
        Self {
            workspace: WorkspaceConfigInput::from_toml(&toml, error_sink),
            local: LocalConfigInput::from_toml(&toml, error_sink),
        }
    }
}

fn get_field_json<T: DeserializeOwned>(
    json: &mut serde_json::Value,
    error_sink: &mut Vec<(String, serde_json::Error)>,
    field: &'static str,
    alias: Option<&'static str>,
) -> Option<T> {
    // XXX: check alias first, to work around the VS Code where it pre-fills the
    // defaults instead of sending an empty object.
    alias
        .into_iter()
        .chain(iter::once(field))
        .filter_map(move |field| {
            let mut pointer = field.replace('_', "/");
            pointer.insert(0, '/');
            json.pointer_mut(&pointer)
                .map(|it| serde_json::from_value(it.take()).map_err(|e| (e, pointer)))
        })
        .flat_map(|res| match res {
            Ok(it) => Some(it),
            Err((e, pointer)) => {
                tracing::warn!("Failed to deserialize config field at {}: {:?}", pointer, e);
                error_sink.push((pointer, e));
                None
            }
        })
        .next()
}

fn get_field_toml<T: DeserializeOwned>(
    toml: &toml::Table,
    error_sink: &mut Vec<(String, toml::de::Error)>,
    field: &'static str,
    alias: Option<&'static str>,
) -> Option<T> {
    // XXX: check alias first, to work around the VS Code where it pre-fills the
    // defaults instead of sending an empty object.
    alias
        .into_iter()
        .chain(iter::once(field))
        .filter_map(move |field| {
            let mut pointer = field.replace('_', "/");
            pointer.insert(0, '/');
            toml_pointer(toml, &pointer)
                .map(|it| <_>::deserialize(it.clone()).map_err(|e| (e, pointer)))
        })
        .find(Result::is_ok)
        .and_then(|res| match res {
            Ok(it) => Some(it),
            Err((e, pointer)) => {
                tracing::warn!("Failed to deserialize config field at {}: {:?}", pointer, e);
                error_sink.push((pointer, e));
                None
            }
        })
}

fn toml_pointer<'a>(toml: &'a toml::Table, pointer: &str) -> Option<&'a toml::Value> {
    fn parse_index(s: &str) -> Option<usize> {
        if s.starts_with('+') || (s.starts_with('0') && s.len() != 1) {
            return None;
        }
        s.parse().ok()
    }

    if pointer.is_empty() {
        return None;
    }
    if !pointer.starts_with('/') {
        return None;
    }
    let mut parts = pointer.split('/').skip(1);
    let first = parts.next()?;
    let init = toml.get(first)?;
    parts.map(|x| x.replace("~1", "/").replace("~0", "~")).try_fold(init, |target, token| {
        match target {
            toml::Value::Table(table) => table.get(&token),
            toml::Value::Array(list) => parse_index(&token).and_then(move |x| list.get(x)),
            _ => None,
        }
    })
}

type SchemaField = (&'static str, &'static str, &'static [&'static str], String);

fn schema(fields: &[SchemaField]) -> serde_json::Value {
    let map = fields
        .iter()
        .map(|(field, ty, doc, default)| {
            let name = field.replace('_', ".");
            let category =
                name.find('.').map(|end| String::from(&name[..end])).unwrap_or("general".into());
            let name = format!("rust-analyzer.{name}");
            let props = field_props(field, ty, doc, default);
            serde_json::json!({
                "title": category,
                "properties": {
                    name: props
                }
            })
        })
        .collect::<Vec<_>>();
    map.into()
}

fn field_props(field: &str, ty: &str, doc: &[&str], default: &str) -> serde_json::Value {
    let doc = doc_comment_to_string(doc);
    let doc = doc.trim_end_matches('\n');
    assert!(
        doc.ends_with('.') && doc.starts_with(char::is_uppercase),
        "bad docs for {field}: {doc:?}"
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
        "Vec<Utf8PathBuf>" => set! {
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
        "FxIndexMap<String, SnippetDef>" => set! {
            "type": "object",
        },
        "FxHashMap<String, String>" => set! {
            "type": "object",
        },
        "FxHashMap<Box<str>, u16>" => set! {
            "type": "object",
        },
        "FxHashMap<String, Option<String>>" => set! {
            "type": "object",
        },
        "Option<usize>" => set! {
            "type": ["null", "integer"],
            "minimum": 0,
        },
        "Option<u16>" => set! {
            "type": ["null", "integer"],
            "minimum": 0,
            "maximum": 65535,
        },
        "Option<String>" => set! {
            "type": ["null", "string"],
        },
        "Option<Utf8PathBuf>" => set! {
            "type": ["null", "string"],
        },
        "Option<bool>" => set! {
            "type": ["null", "boolean"],
        },
        "Option<Vec<String>>" => set! {
            "type": ["null", "array"],
            "items": { "type": "string" },
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
            "enum": ["preserve", "crate", "module", "item", "one"],
            "enumDescriptions": [
                "Do not change the granularity of any imports and preserve the original structure written by the developer.",
                "Merge imports from the same crate into a single use statement. Conversely, imports from different crates are split into separate statements.",
                "Merge imports from the same module into a single use statement. Conversely, imports from different modules are split into separate statements.",
                "Flatten imports so that each has its own use statement.",
                "Merge all imports into a single use statement as long as they have the same visibility and attributes."
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
        "AdjustmentHintsDef" => set! {
            "type": "string",
            "enum": [
                "always",
                "never",
                "reborrow"
            ],
            "enumDescriptions": [
                "Always show all adjustment hints.",
                "Never show adjustment hints.",
                "Only show auto borrow and dereference adjustment hints."
            ]
        },
        "DiscriminantHintsDef" => set! {
            "type": "string",
            "enum": [
                "always",
                "never",
                "fieldless"
            ],
            "enumDescriptions": [
                "Always show all discriminant hints.",
                "Never show discriminant hints.",
                "Only show discriminant hints on fieldless enum variants."
            ]
        },
        "AdjustmentHintsModeDef" => set! {
            "type": "string",
            "enum": [
                "prefix",
                "postfix",
                "prefer_prefix",
                "prefer_postfix",
            ],
            "enumDescriptions": [
                "Always show adjustment hints as prefix (`*expr`).",
                "Always show adjustment hints as postfix (`expr.*`).",
                "Show prefix or postfix depending on which uses less parenthesis, preferring prefix.",
                "Show prefix or postfix depending on which uses less parenthesis, preferring postfix.",
            ]
        },
        "CargoFeaturesDef" => set! {
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
        "Option<CargoFeaturesDef>" => set! {
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
        "AnnotationLocation" => set! {
            "type": "string",
            "enum": ["above_name", "above_whole_item"],
            "enumDescriptions": [
                "Render annotations above the name of the item.",
                "Render annotations above the whole item, including documentation comments and attributes."
            ],
        },
        "InvocationStrategy" => set! {
            "type": "string",
            "enum": ["per_workspace", "once"],
            "enumDescriptions": [
                "The command will be executed for each Rust workspace with the workspace as the working directory.",
                "The command will be executed once with the opened project as the working directory."
            ],
        },
        "Option<CheckOnSaveTargets>" => set! {
            "anyOf": [
                {
                    "type": "null"
                },
                {
                    "type": "string",
                },
                {
                    "type": "array",
                    "items": { "type": "string" }
                },
            ],
        },
        "ClosureStyle" => set! {
            "type": "string",
            "enum": ["impl_fn", "rust_analyzer", "with_id", "hide"],
            "enumDescriptions": [
                "`impl_fn`: `impl FnMut(i32, u64) -> i8`",
                "`rust_analyzer`: `|i32, u64| -> i8`",
                "`with_id`: `{closure#14352}`, where that id is the unique number of the closure in r-a internals",
                "`hide`: Shows `...` for every closure type",
            ],
        },
        "Option<MemoryLayoutHoverRenderKindDef>" => set! {
            "anyOf": [
                {
                    "type": "null"
                },
                {
                    "type": "string",
                    "enum": ["both", "decimal", "hexadecimal", ],
                    "enumDescriptions": [
                        "Render as 12 (0xC)",
                        "Render as 12",
                        "Render as 0xC"
                    ],
                },
            ],
        },
        "Option<TargetDirectory>" => set! {
            "anyOf": [
                {
                    "type": "null"
                },
                {
                    "type": "boolean"
                },
                {
                    "type": "string"
                },
            ],
        },
        "NumThreads" => set! {
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 255
                },
                {
                    "type": "string",
                    "enum": ["physical", "logical", ],
                    "enumDescriptions": [
                        "Use the number of physical cores",
                        "Use the number of logical cores",
                    ],
                },
            ],
        },
        "Option<NumThreads>" => set! {
            "anyOf": [
                {
                    "type": "null"
                },
                {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 255
                },
                {
                    "type": "string",
                    "enum": ["physical", "logical", ],
                    "enumDescriptions": [
                        "Use the number of physical cores",
                        "Use the number of logical cores",
                    ],
                },
            ],
        },
        "Option<DiscoverWorkspaceConfig>" => set! {
            "anyOf": [
                {
                    "type": "null"
                },
                {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "progressLabel": {
                            "type": "string"
                        },
                        "filesToWatch": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                    }
                }
            ]
        },
        "Option<MaxSubstitutionLength>" => set! {
            "anyOf": [
                {
                    "type": "null"
                },
                {
                    "type": "string",
                    "enum": ["hide"]
                },
                {
                    "type": "integer"
                }
            ]
        },
        "Vec<AutoImportExclusion>" => set! {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "string",
                    },
                    {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["always", "methods"],
                                "enumDescriptions": [
                                    "Do not show this item or its methods (if it is a trait) in auto-import completions.",
                                    "Do not show this traits methods in auto-import completions."
                                ],
                            },
                        }
                    }
                ]
             }
        },
        _ => panic!("missing entry for {ty}: {default} (field {field})"),
    }

    map.into()
}

fn validate_toml_table(
    known_ptrs: &[&[&'static str]],
    toml: &toml::Table,
    ptr: &mut String,
    error_sink: &mut Vec<(String, toml::de::Error)>,
) {
    let verify = |ptr: &String| known_ptrs.iter().any(|ptrs| ptrs.contains(&ptr.as_str()));

    let l = ptr.len();
    for (k, v) in toml {
        if !ptr.is_empty() {
            ptr.push('_');
        }
        ptr.push_str(k);

        match v {
            // This is a table config, any entry in it is therefore valid
            toml::Value::Table(_) if verify(ptr) => (),
            toml::Value::Table(table) => validate_toml_table(known_ptrs, table, ptr, error_sink),
            _ if !verify(ptr) => error_sink
                .push((ptr.replace('_', "/"), toml::de::Error::custom("unexpected field"))),
            _ => (),
        }

        ptr.truncate(l);
    }
}

#[cfg(test)]
fn manual(fields: &[SchemaField]) -> String {
    fields.iter().fold(String::new(), |mut acc, (field, _ty, doc, default)| {
        let id = field.replace('_', ".");
        let name = format!("rust-analyzer.{id}");
        let doc = doc_comment_to_string(doc);
        if default.contains('\n') {
            format_to_acc!(
                acc,
                "## {name} {{#{id}}}\n\nDefault:\n```json\n{default}\n```\n\n{doc}\n\n"
            )
        } else {
            format_to_acc!(acc, "## {name} {{#{id}}}\n\nDefault: `{default}`\n\n{doc}\n\n")
        }
    })
}

fn doc_comment_to_string(doc: &[&str]) -> String {
    doc.iter()
        .map(|it| it.strip_prefix(' ').unwrap_or(it))
        .fold(String::new(), |mut acc, it| format_to_acc!(acc, "{it}\n"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use test_utils::{ensure_file_contents, project_root};

    use super::*;

    #[test]
    fn generate_package_json_config() {
        let s = Config::json_schema();

        let schema = format!("{s:#}");
        let mut schema = schema
            .trim_start_matches('[')
            .trim_end_matches(']')
            .replace("  ", "    ")
            .replace('\n', "\n        ")
            .trim_start_matches('\n')
            .trim_end()
            .to_owned();
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
            if let Some(link_end) = link.find([' ', '[']) {
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

        let start_marker =
            "            {\n                \"title\": \"$generated-start\"\n            },\n";
        let end_marker =
            "            {\n                \"title\": \"$generated-end\"\n            }\n";

        let start = package_json.find(start_marker).unwrap() + start_marker.len();
        let end = package_json.find(end_marker).unwrap();

        let p = remove_ws(&package_json[start..end]);
        let s = remove_ws(&schema);
        if !p.contains(&s) {
            package_json.replace_range(start..end, &schema);
            ensure_file_contents(package_json_path.as_std_path(), &package_json)
        }
    }

    #[test]
    fn generate_config_documentation() {
        let docs_path = project_root().join("docs/book/src/configuration_generated.md");
        let expected = FullConfigInput::manual();
        ensure_file_contents(docs_path.as_std_path(), &expected);
    }

    fn remove_ws(text: &str) -> String {
        text.replace(char::is_whitespace, "")
    }

    #[test]
    fn proc_macro_srv_null() {
        let mut config =
            Config::new(AbsPathBuf::assert(project_root()), Default::default(), vec![], None);

        let mut change = ConfigChange::default();
        change.change_client_config(serde_json::json!({
            "procMacro" : {
                "server": null,
        }}));

        (config, _, _) = config.apply_change(change);
        assert_eq!(config.proc_macro_srv(), None);
    }

    #[test]
    fn proc_macro_srv_abs() {
        let mut config =
            Config::new(AbsPathBuf::assert(project_root()), Default::default(), vec![], None);
        let mut change = ConfigChange::default();
        change.change_client_config(serde_json::json!({
        "procMacro" : {
            "server": project_root().to_string(),
        }}));

        (config, _, _) = config.apply_change(change);
        assert_eq!(config.proc_macro_srv(), Some(AbsPathBuf::assert(project_root())));
    }

    #[test]
    fn proc_macro_srv_rel() {
        let mut config =
            Config::new(AbsPathBuf::assert(project_root()), Default::default(), vec![], None);

        let mut change = ConfigChange::default();

        change.change_client_config(serde_json::json!({
        "procMacro" : {
            "server": "./server"
        }}));

        (config, _, _) = config.apply_change(change);

        assert_eq!(
            config.proc_macro_srv(),
            Some(AbsPathBuf::try_from(project_root().join("./server")).unwrap())
        );
    }

    #[test]
    fn cargo_target_dir_unset() {
        let mut config =
            Config::new(AbsPathBuf::assert(project_root()), Default::default(), vec![], None);

        let mut change = ConfigChange::default();

        change.change_client_config(serde_json::json!({
            "rust" : { "analyzerTargetDir" : null }
        }));

        (config, _, _) = config.apply_change(change);
        assert_eq!(config.cargo_targetDir(None), &None);
        assert!(
            matches!(config.flycheck(None), FlycheckConfig::CargoCommand { options, .. } if options.target_dir.is_none())
        );
    }

    #[test]
    fn cargo_target_dir_subdir() {
        let mut config =
            Config::new(AbsPathBuf::assert(project_root()), Default::default(), vec![], None);

        let mut change = ConfigChange::default();
        change.change_client_config(serde_json::json!({
            "rust" : { "analyzerTargetDir" : true }
        }));

        (config, _, _) = config.apply_change(change);

        assert_eq!(config.cargo_targetDir(None), &Some(TargetDirectory::UseSubdirectory(true)));
        let target =
            Utf8PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or("target".to_owned()));
        assert!(
            matches!(config.flycheck(None), FlycheckConfig::CargoCommand { options, .. } if options.target_dir == Some(target.join("rust-analyzer")))
        );
    }

    #[test]
    fn cargo_target_dir_relative_dir() {
        let mut config =
            Config::new(AbsPathBuf::assert(project_root()), Default::default(), vec![], None);

        let mut change = ConfigChange::default();
        change.change_client_config(serde_json::json!({
            "rust" : { "analyzerTargetDir" : "other_folder" }
        }));

        (config, _, _) = config.apply_change(change);

        assert_eq!(
            config.cargo_targetDir(None),
            &Some(TargetDirectory::Directory(Utf8PathBuf::from("other_folder")))
        );
        assert!(
            matches!(config.flycheck(None), FlycheckConfig::CargoCommand { options, .. } if options.target_dir == Some(Utf8PathBuf::from("other_folder")))
        );
    }
}
