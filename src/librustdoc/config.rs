use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fmt, io};

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::DiagCtxtHandle;
use rustc_session::config::{
    self, CodegenOptions, CrateType, ErrorOutputType, Externs, Input, JsonUnusedExterns,
    OptionsTargetModifiers, OutFileName, Sysroot, UnstableOptions, get_cmd_lint_options,
    nightly_options, parse_crate_types_from_list, parse_externs, parse_target_triple,
};
use rustc_session::lint::Level;
use rustc_session::search_paths::SearchPath;
use rustc_session::{EarlyDiagCtxt, getopts};
use rustc_span::FileName;
use rustc_span::edition::Edition;
use rustc_target::spec::TargetTuple;

use crate::core::new_dcx;
use crate::externalfiles::ExternalHtml;
use crate::html::markdown::IdMap;
use crate::html::render::StylePath;
use crate::html::static_files;
use crate::passes::{self, Condition};
use crate::scrape_examples::{AllCallLocations, ScrapeExamplesOptions};
use crate::{html, opts, theme};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum OutputFormat {
    Json,
    #[default]
    Html,
    Doctest,
}

impl OutputFormat {
    pub(crate) fn is_json(&self) -> bool {
        matches!(self, OutputFormat::Json)
    }
}

impl TryFrom<&str> for OutputFormat {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "json" => Ok(OutputFormat::Json),
            "html" => Ok(OutputFormat::Html),
            "doctest" => Ok(OutputFormat::Doctest),
            _ => Err(format!("unknown output format `{value}`")),
        }
    }
}

/// Either an input crate, markdown file, or nothing (--merge=finalize).
pub(crate) enum InputMode {
    /// The `--merge=finalize` step does not need an input crate to rustdoc.
    NoInputMergeFinalize,
    /// A crate or markdown file.
    HasFile(Input),
}

/// Configuration options for rustdoc.
#[derive(Clone)]
pub(crate) struct Options {
    // Basic options / Options passed directly to rustc
    /// The name of the crate being documented.
    pub(crate) crate_name: Option<String>,
    /// Whether or not this is a bin crate
    pub(crate) bin_crate: bool,
    /// Whether or not this is a proc-macro crate
    pub(crate) proc_macro_crate: bool,
    /// How to format errors and warnings.
    pub(crate) error_format: ErrorOutputType,
    /// Width of output buffer to truncate errors appropriately.
    pub(crate) diagnostic_width: Option<usize>,
    /// Library search paths to hand to the compiler.
    pub(crate) libs: Vec<SearchPath>,
    /// Library search paths strings to hand to the compiler.
    pub(crate) lib_strs: Vec<String>,
    /// The list of external crates to link against.
    pub(crate) externs: Externs,
    /// The list of external crates strings to link against.
    pub(crate) extern_strs: Vec<String>,
    /// List of `cfg` flags to hand to the compiler. Always includes `rustdoc`.
    pub(crate) cfgs: Vec<String>,
    /// List of check cfg flags to hand to the compiler.
    pub(crate) check_cfgs: Vec<String>,
    /// Codegen options to hand to the compiler.
    pub(crate) codegen_options: CodegenOptions,
    /// Codegen options strings to hand to the compiler.
    pub(crate) codegen_options_strs: Vec<String>,
    /// Unstable (`-Z`) options to pass to the compiler.
    pub(crate) unstable_opts: UnstableOptions,
    /// Unstable (`-Z`) options strings to pass to the compiler.
    pub(crate) unstable_opts_strs: Vec<String>,
    /// The target used to compile the crate against.
    pub(crate) target: TargetTuple,
    /// Edition used when reading the crate. Defaults to "2015". Also used by default when
    /// compiling doctests from the crate.
    pub(crate) edition: Edition,
    /// The path to the sysroot. Used during the compilation process.
    pub(crate) sysroot: Sysroot,
    /// Lint information passed over the command-line.
    pub(crate) lint_opts: Vec<(String, Level)>,
    /// Whether to ask rustc to describe the lints it knows.
    pub(crate) describe_lints: bool,
    /// What level to cap lints at.
    pub(crate) lint_cap: Option<Level>,

    // Options specific to running doctests
    /// Whether we should run doctests instead of generating docs.
    pub(crate) should_test: bool,
    /// List of arguments to pass to the test harness, if running tests.
    pub(crate) test_args: Vec<String>,
    /// The working directory in which to run tests.
    pub(crate) test_run_directory: Option<PathBuf>,
    /// Optional path to persist the doctest executables to, defaults to a
    /// temporary directory if not set.
    pub(crate) persist_doctests: Option<PathBuf>,
    /// Runtool to run doctests with
    pub(crate) test_runtool: Option<String>,
    /// Arguments to pass to the runtool
    pub(crate) test_runtool_args: Vec<String>,
    /// Do not run doctests, compile them if should_test is active.
    pub(crate) no_run: bool,
    /// What sources are being mapped.
    pub(crate) remap_path_prefix: Vec<(PathBuf, PathBuf)>,

    /// The path to a rustc-like binary to build tests with. If not set, we
    /// default to loading from `$sysroot/bin/rustc`.
    pub(crate) test_builder: Option<PathBuf>,

    /// Run these wrapper instead of rustc directly
    pub(crate) test_builder_wrappers: Vec<PathBuf>,

    // Options that affect the documentation process
    /// Whether to run the `calculate-doc-coverage` pass, which counts the number of public items
    /// with and without documentation.
    pub(crate) show_coverage: bool,

    // Options that alter generated documentation pages
    /// Crate version to note on the sidebar of generated docs.
    pub(crate) crate_version: Option<String>,
    /// The format that we output when rendering.
    ///
    /// Currently used only for the `--show-coverage` option.
    pub(crate) output_format: OutputFormat,
    /// If this option is set to `true`, rustdoc will only run checks and not generate
    /// documentation.
    pub(crate) run_check: bool,
    /// Whether doctests should emit unused externs
    pub(crate) json_unused_externs: JsonUnusedExterns,
    /// Whether to skip capturing stdout and stderr of tests.
    pub(crate) no_capture: bool,

    /// Configuration for scraping examples from the current crate. If this option is Some(..) then
    /// the compiler will scrape examples and not generate documentation.
    pub(crate) scrape_examples_options: Option<ScrapeExamplesOptions>,

    /// Note: this field is duplicated in `RenderOptions` because it's useful
    /// to have it in both places.
    pub(crate) unstable_features: rustc_feature::UnstableFeatures,

    /// Arguments to be used when compiling doctests.
    pub(crate) doctest_build_args: Vec<String>,

    /// Target modifiers.
    pub(crate) target_modifiers: BTreeMap<OptionsTargetModifiers, String>,
}

impl fmt::Debug for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct FmtExterns<'a>(&'a Externs);

        impl fmt::Debug for FmtExterns<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_map().entries(self.0.iter()).finish()
            }
        }

        f.debug_struct("Options")
            .field("crate_name", &self.crate_name)
            .field("bin_crate", &self.bin_crate)
            .field("proc_macro_crate", &self.proc_macro_crate)
            .field("error_format", &self.error_format)
            .field("libs", &self.libs)
            .field("externs", &FmtExterns(&self.externs))
            .field("cfgs", &self.cfgs)
            .field("check-cfgs", &self.check_cfgs)
            .field("codegen_options", &"...")
            .field("unstable_options", &"...")
            .field("target", &self.target)
            .field("edition", &self.edition)
            .field("sysroot", &self.sysroot)
            .field("lint_opts", &self.lint_opts)
            .field("describe_lints", &self.describe_lints)
            .field("lint_cap", &self.lint_cap)
            .field("should_test", &self.should_test)
            .field("test_args", &self.test_args)
            .field("test_run_directory", &self.test_run_directory)
            .field("persist_doctests", &self.persist_doctests)
            .field("show_coverage", &self.show_coverage)
            .field("crate_version", &self.crate_version)
            .field("test_runtool", &self.test_runtool)
            .field("test_runtool_args", &self.test_runtool_args)
            .field("run_check", &self.run_check)
            .field("no_run", &self.no_run)
            .field("test_builder_wrappers", &self.test_builder_wrappers)
            .field("remap-file-prefix", &self.remap_path_prefix)
            .field("no_capture", &self.no_capture)
            .field("scrape_examples_options", &self.scrape_examples_options)
            .field("unstable_features", &self.unstable_features)
            .finish()
    }
}

/// Configuration options for the HTML page-creation process.
#[derive(Clone, Debug)]
pub(crate) struct RenderOptions {
    /// Output directory to generate docs into. Defaults to `doc`.
    pub(crate) output: PathBuf,
    /// External files to insert into generated pages.
    pub(crate) external_html: ExternalHtml,
    /// A pre-populated `IdMap` with the default headings and any headings added by Markdown files
    /// processed by `external_html`.
    pub(crate) id_map: IdMap,
    /// If present, playground URL to use in the "Run" button added to code samples.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    pub(crate) playground_url: Option<String>,
    /// What sorting mode to use for module pages.
    /// `ModuleSorting::Alphabetical` by default.
    pub(crate) module_sorting: ModuleSorting,
    /// List of themes to extend the docs with. Original argument name is included to assist in
    /// displaying errors if it fails a theme check.
    pub(crate) themes: Vec<StylePath>,
    /// If present, CSS file that contains rules to add to the default CSS.
    pub(crate) extension_css: Option<PathBuf>,
    /// A map of crate names to the URL to use instead of querying the crate's `html_root_url`.
    pub(crate) extern_html_root_urls: BTreeMap<String, String>,
    /// Whether to give precedence to `html_root_url` or `--extern-html-root-url`.
    pub(crate) extern_html_root_takes_precedence: bool,
    /// A map of the default settings (values are as for DOM storage API). Keys should lack the
    /// `rustdoc-` prefix.
    pub(crate) default_settings: FxIndexMap<String, String>,
    /// If present, suffix added to CSS/JavaScript files when referencing them in generated pages.
    pub(crate) resource_suffix: String,
    /// Whether to create an index page in the root of the output directory. If this is true but
    /// `enable_index_page` is None, generate a static listing of crates instead.
    pub(crate) enable_index_page: bool,
    /// A file to use as the index page at the root of the output directory. Overrides
    /// `enable_index_page` to be true if set.
    pub(crate) index_page: Option<PathBuf>,
    /// An optional path to use as the location of static files. If not set, uses combinations of
    /// `../` to reach the documentation root.
    pub(crate) static_root_path: Option<String>,

    // Options specific to reading standalone Markdown files
    /// Whether to generate a table of contents on the output file when reading a standalone
    /// Markdown file.
    pub(crate) markdown_no_toc: bool,
    /// Additional CSS files to link in pages generated from standalone Markdown files.
    pub(crate) markdown_css: Vec<String>,
    /// If present, playground URL to use in the "Run" button added to code samples generated from
    /// standalone Markdown files. If not present, `playground_url` is used.
    pub(crate) markdown_playground_url: Option<String>,
    /// Document items that have lower than `pub` visibility.
    pub(crate) document_private: bool,
    /// Document items that have `doc(hidden)`.
    pub(crate) document_hidden: bool,
    /// If `true`, generate a JSON file in the crate folder instead of HTML redirection files.
    pub(crate) generate_redirect_map: bool,
    /// Show the memory layout of types in the docs.
    pub(crate) show_type_layout: bool,
    /// Note: this field is duplicated in `Options` because it's useful to have
    /// it in both places.
    pub(crate) unstable_features: rustc_feature::UnstableFeatures,
    pub(crate) emit: Vec<EmitType>,
    /// If `true`, HTML source pages will generate links for items to their definition.
    pub(crate) generate_link_to_definition: bool,
    /// Set of function-call locations to include as examples
    pub(crate) call_locations: AllCallLocations,
    /// If `true`, Context::init will not emit shared files.
    pub(crate) no_emit_shared: bool,
    /// If `true`, HTML source code pages won't be generated.
    pub(crate) html_no_source: bool,
    /// This field is only used for the JSON output. If it's set to true, no file will be created
    /// and content will be displayed in stdout directly.
    pub(crate) output_to_stdout: bool,
    /// Whether we should read or write rendered cross-crate info in the doc root.
    pub(crate) should_merge: ShouldMerge,
    /// Path to crate-info for external crates.
    pub(crate) include_parts_dir: Vec<PathToParts>,
    /// Where to write crate-info
    pub(crate) parts_out_dir: Option<PathToParts>,
    /// disable minification of CSS/JS
    pub(crate) disable_minification: bool,
    /// If `true`, HTML source pages will generate the possibility to expand macros.
    pub(crate) generate_macro_expansion: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModuleSorting {
    DeclarationOrder,
    Alphabetical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EmitType {
    Toolchain,
    InvocationSpecific,
    DepInfo(Option<OutFileName>),
}

impl FromStr for EmitType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "toolchain-shared-resources" => Ok(Self::Toolchain),
            "invocation-specific" => Ok(Self::InvocationSpecific),
            "dep-info" => Ok(Self::DepInfo(None)),
            option => match option.strip_prefix("dep-info=") {
                Some("-") => Ok(Self::DepInfo(Some(OutFileName::Stdout))),
                Some(f) => Ok(Self::DepInfo(Some(OutFileName::Real(f.into())))),
                None => Err(()),
            },
        }
    }
}

impl RenderOptions {
    pub(crate) fn should_emit_crate(&self) -> bool {
        self.emit.is_empty() || self.emit.contains(&EmitType::InvocationSpecific)
    }

    pub(crate) fn dep_info(&self) -> Option<Option<&OutFileName>> {
        for emit in &self.emit {
            if let EmitType::DepInfo(file) = emit {
                return Some(file.as_ref());
            }
        }
        None
    }
}

/// Create the input (string or file path)
///
/// Warning: Return an unrecoverable error in case of error!
fn make_input(early_dcx: &EarlyDiagCtxt, input: &str) -> Input {
    if input == "-" {
        let mut src = String::new();
        if io::stdin().read_to_string(&mut src).is_err() {
            // Immediately stop compilation if there was an issue reading
            // the input (for example if the input stream is not UTF-8).
            early_dcx.early_fatal("couldn't read from stdin, as it did not contain valid UTF-8");
        }
        Input::Str { name: FileName::anon_source_code(&src), input: src }
    } else {
        Input::File(PathBuf::from(input))
    }
}

impl Options {
    /// Parses the given command-line for options. If an error message or other early-return has
    /// been printed, returns `Err` with the exit code.
    pub(crate) fn from_matches(
        early_dcx: &mut EarlyDiagCtxt,
        matches: &getopts::Matches,
        args: Vec<String>,
    ) -> Option<(InputMode, Options, RenderOptions, Vec<PathBuf>)> {
        // Check for unstable options.
        nightly_options::check_nightly_options(early_dcx, matches, &opts());

        if args.is_empty() || matches.opt_present("h") || matches.opt_present("help") {
            crate::usage("rustdoc");
            return None;
        } else if matches.opt_present("version") {
            rustc_driver::version!(&early_dcx, "rustdoc", matches);
            return None;
        }

        if rustc_driver::describe_flag_categories(early_dcx, matches) {
            return None;
        }

        let color = config::parse_color(early_dcx, matches);
        let crate_name = matches.opt_str("crate-name");
        let unstable_features =
            rustc_feature::UnstableFeatures::from_environment(crate_name.as_deref());
        let config::JsonConfig { json_rendered, json_unused_externs, json_color, .. } =
            config::parse_json(early_dcx, matches, unstable_features.is_nightly_build());
        let error_format = config::parse_error_format(
            early_dcx,
            matches,
            color,
            json_color,
            json_rendered,
            unstable_features.is_nightly_build(),
        );
        let diagnostic_width = matches.opt_get("diagnostic-width").unwrap_or_default();

        let mut target_modifiers = BTreeMap::<OptionsTargetModifiers, String>::new();
        let codegen_options = CodegenOptions::build(early_dcx, matches, &mut target_modifiers);
        let unstable_opts = UnstableOptions::build(early_dcx, matches, &mut target_modifiers);

        let remap_path_prefix = match parse_remap_path_prefix(matches) {
            Ok(prefix_mappings) => prefix_mappings,
            Err(err) => {
                early_dcx.early_fatal(err);
            }
        };

        let dcx = new_dcx(error_format, None, diagnostic_width, &unstable_opts);
        let dcx = dcx.handle();

        // check for deprecated options
        check_deprecated_options(matches, dcx);

        if matches.opt_strs("passes") == ["list"] {
            println!("Available passes for running rustdoc:");
            for pass in passes::PASSES {
                println!("{:>20} - {}", pass.name, pass.description);
            }
            println!("\nDefault passes for rustdoc:");
            for p in passes::DEFAULT_PASSES {
                print!("{:>20}", p.pass.name);
                println_condition(p.condition);
            }

            if nightly_options::match_is_nightly_build(matches) {
                println!("\nPasses run with `--show-coverage`:");
                for p in passes::COVERAGE_PASSES {
                    print!("{:>20}", p.pass.name);
                    println_condition(p.condition);
                }
            }

            fn println_condition(condition: Condition) {
                use Condition::*;
                match condition {
                    Always => println!(),
                    WhenDocumentPrivate => println!("  (when --document-private-items)"),
                    WhenNotDocumentPrivate => println!("  (when not --document-private-items)"),
                    WhenNotDocumentHidden => println!("  (when not --document-hidden-items)"),
                }
            }

            return None;
        }

        let mut emit = FxIndexMap::<_, EmitType>::default();
        for list in matches.opt_strs("emit") {
            for kind in list.split(',') {
                match kind.parse() {
                    Ok(kind) => {
                        // De-duplicate emit types and the last wins.
                        // Only one instance for each type is allowed
                        // regardless the actual data it carries.
                        // This matches rustc's `--emit` behavior.
                        emit.insert(std::mem::discriminant(&kind), kind);
                    }
                    Err(()) => dcx.fatal(format!("unrecognized emission type: {kind}")),
                }
            }
        }
        let emit = emit.into_values().collect::<Vec<_>>();

        let show_coverage = matches.opt_present("show-coverage");
        let output_format_s = matches.opt_str("output-format");
        let output_format = match output_format_s {
            Some(ref s) => match OutputFormat::try_from(s.as_str()) {
                Ok(out_fmt) => out_fmt,
                Err(e) => dcx.fatal(e),
            },
            None => OutputFormat::default(),
        };

        // check for `--output-format=json`
        match (
            output_format_s.as_ref().map(|_| output_format),
            show_coverage,
            nightly_options::is_unstable_enabled(matches),
        ) {
            (None | Some(OutputFormat::Json), true, _) => {}
            (_, true, _) => {
                dcx.fatal(format!(
                    "`--output-format={}` is not supported for the `--show-coverage` option",
                    output_format_s.unwrap_or_default(),
                ));
            }
            // If `-Zunstable-options` is used, nothing to check after this point.
            (_, false, true) => {}
            (None | Some(OutputFormat::Html), false, _) => {}
            (Some(OutputFormat::Json), false, false) => {
                dcx.fatal(
                    "the -Z unstable-options flag must be passed to enable --output-format for documentation generation (see https://github.com/rust-lang/rust/issues/76578)",
                );
            }
            (Some(OutputFormat::Doctest), false, false) => {
                dcx.fatal(
                    "the -Z unstable-options flag must be passed to enable --output-format for documentation generation (see https://github.com/rust-lang/rust/issues/134529)",
                );
            }
        }

        let to_check = matches.opt_strs("check-theme");
        if !to_check.is_empty() {
            let mut content =
                std::str::from_utf8(static_files::STATIC_FILES.rustdoc_css.src_bytes).unwrap();
            if let Some((_, inside)) = content.split_once("/* Begin theme: light */") {
                content = inside;
            }
            if let Some((inside, _)) = content.split_once("/* End theme: light */") {
                content = inside;
            }
            let paths = match theme::load_css_paths(content) {
                Ok(p) => p,
                Err(e) => dcx.fatal(e),
            };
            let mut errors = 0;

            println!("rustdoc: [check-theme] Starting tests! (Ignoring all other arguments)");
            for theme_file in to_check.iter() {
                print!(" - Checking \"{theme_file}\"...");
                let (success, differences) = theme::test_theme_against(theme_file, &paths, dcx);
                if !differences.is_empty() || !success {
                    println!(" FAILED");
                    errors += 1;
                    if !differences.is_empty() {
                        println!("{}", differences.join("\n"));
                    }
                } else {
                    println!(" OK");
                }
            }
            if errors != 0 {
                dcx.fatal("[check-theme] one or more tests failed");
            }
            return None;
        }

        let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(early_dcx, matches);

        let input = if describe_lints {
            InputMode::HasFile(make_input(early_dcx, ""))
        } else {
            match matches.free.as_slice() {
                [] if matches.opt_str("merge").as_deref() == Some("finalize") => {
                    InputMode::NoInputMergeFinalize
                }
                [] => dcx.fatal("missing file operand"),
                [input] => InputMode::HasFile(make_input(early_dcx, input)),
                _ => dcx.fatal("too many file operands"),
            }
        };

        let externs = parse_externs(early_dcx, matches, &unstable_opts);
        let extern_html_root_urls = match parse_extern_html_roots(matches) {
            Ok(ex) => ex,
            Err(err) => dcx.fatal(err),
        };

        let parts_out_dir =
            match matches.opt_str("parts-out-dir").map(PathToParts::from_flag).transpose() {
                Ok(parts_out_dir) => parts_out_dir,
                Err(e) => dcx.fatal(e),
            };
        let include_parts_dir = match parse_include_parts_dir(matches) {
            Ok(include_parts_dir) => include_parts_dir,
            Err(e) => dcx.fatal(e),
        };

        let default_settings: Vec<Vec<(String, String)>> = vec![
            matches
                .opt_str("default-theme")
                .iter()
                .flat_map(|theme| {
                    vec![
                        ("use-system-theme".to_string(), "false".to_string()),
                        ("theme".to_string(), theme.to_string()),
                    ]
                })
                .collect(),
            matches
                .opt_strs("default-setting")
                .iter()
                .map(|s| match s.split_once('=') {
                    None => (s.clone(), "true".to_string()),
                    Some((k, v)) => (k.to_string(), v.to_string()),
                })
                .collect(),
        ];
        let default_settings = default_settings
            .into_iter()
            .flatten()
            .map(
                // The keys here become part of `data-` attribute names in the generated HTML.  The
                // browser does a strange mapping when converting them into attributes on the
                // `dataset` property on the DOM HTML Node:
                //   https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/dataset
                //
                // The original key values we have are the same as the DOM storage API keys and the
                // command line options, so contain `-`.  Our JavaScript needs to be able to look
                // these values up both in `dataset` and in the storage API, so it needs to be able
                // to convert the names back and forth.  Despite doing this kebab-case to
                // StudlyCaps transformation automatically, the JS DOM API does not provide a
                // mechanism for doing just the transformation on a string.  So we want to avoid
                // the StudlyCaps representation in the `dataset` property.
                //
                // We solve this by replacing all the `-`s with `_`s.  We do that here, when we
                // generate the `data-` attributes, and in the JS, when we look them up.  (See
                // `getSettingValue` in `storage.js.`) Converting `-` to `_` is simple in JS.
                //
                // The values will be HTML-escaped by the default Tera escaping.
                |(k, v)| (k.replace('-', "_"), v),
            )
            .collect();

        let test_args = matches.opt_strs("test-args");
        let test_args: Vec<String> =
            test_args.iter().flat_map(|s| s.split_whitespace()).map(|s| s.to_string()).collect();

        let should_test = matches.opt_present("test");
        let no_run = matches.opt_present("no-run");

        if !should_test && no_run {
            dcx.fatal("the `--test` flag must be passed to enable `--no-run`");
        }

        let mut output_to_stdout = false;
        let test_builder_wrappers =
            matches.opt_strs("test-builder-wrapper").iter().map(PathBuf::from).collect();
        let output = match (matches.opt_str("out-dir"), matches.opt_str("output")) {
            (Some(_), Some(_)) => {
                dcx.fatal("cannot use both 'out-dir' and 'output' at once");
            }
            (Some(out_dir), None) | (None, Some(out_dir)) => {
                output_to_stdout = out_dir == "-";
                PathBuf::from(out_dir)
            }
            (None, None) => PathBuf::from("doc"),
        };

        let cfgs = matches.opt_strs("cfg");
        let check_cfgs = matches.opt_strs("check-cfg");

        let extension_css = matches.opt_str("e").map(|s| PathBuf::from(&s));

        let mut loaded_paths = Vec::new();

        if let Some(ref p) = extension_css {
            loaded_paths.push(p.clone());
            if !p.is_file() {
                dcx.fatal("option --extend-css argument must be a file");
            }
        }

        let mut themes = Vec::new();
        if matches.opt_present("theme") {
            let mut content =
                std::str::from_utf8(static_files::STATIC_FILES.rustdoc_css.src_bytes).unwrap();
            if let Some((_, inside)) = content.split_once("/* Begin theme: light */") {
                content = inside;
            }
            if let Some((inside, _)) = content.split_once("/* End theme: light */") {
                content = inside;
            }
            let paths = match theme::load_css_paths(content) {
                Ok(p) => p,
                Err(e) => dcx.fatal(e),
            };

            for (theme_file, theme_s) in
                matches.opt_strs("theme").iter().map(|s| (PathBuf::from(&s), s.to_owned()))
            {
                if !theme_file.is_file() {
                    dcx.struct_fatal(format!("invalid argument: \"{theme_s}\""))
                        .with_help("arguments to --theme must be files")
                        .emit();
                }
                if theme_file.extension() != Some(OsStr::new("css")) {
                    dcx.struct_fatal(format!("invalid argument: \"{theme_s}\""))
                        .with_help("arguments to --theme must have a .css extension")
                        .emit();
                }
                let (success, ret) = theme::test_theme_against(&theme_file, &paths, dcx);
                if !success {
                    dcx.fatal(format!("error loading theme file: \"{theme_s}\""));
                } else if !ret.is_empty() {
                    dcx.struct_warn(format!(
                        "theme file \"{theme_s}\" is missing CSS rules from the default theme",
                    ))
                    .with_warn("the theme may appear incorrect when loaded")
                    .with_help(format!(
                        "to see what rules are missing, call `rustdoc --check-theme \"{theme_s}\"`",
                    ))
                    .emit();
                }
                loaded_paths.push(theme_file.clone());
                themes.push(StylePath { path: theme_file });
            }
        }

        let edition = config::parse_crate_edition(early_dcx, matches);

        let mut id_map = html::markdown::IdMap::new();
        let Some(external_html) = ExternalHtml::load(
            &matches.opt_strs("html-in-header"),
            &matches.opt_strs("html-before-content"),
            &matches.opt_strs("html-after-content"),
            &matches.opt_strs("markdown-before-content"),
            &matches.opt_strs("markdown-after-content"),
            nightly_options::match_is_nightly_build(matches),
            dcx,
            &mut id_map,
            edition,
            &None,
            &mut loaded_paths,
        ) else {
            dcx.fatal("`ExternalHtml::load` failed");
        };

        match matches.opt_str("r").as_deref() {
            Some("rust") | None => {}
            Some(s) => dcx.fatal(format!("unknown input format: {s}")),
        }

        let index_page = matches.opt_str("index-page").map(|s| PathBuf::from(&s));
        if let Some(ref index_page) = index_page
            && !index_page.is_file()
        {
            dcx.fatal("option `--index-page` argument must be a file");
        }

        let target = parse_target_triple(early_dcx, matches);
        let sysroot = Sysroot::new(matches.opt_str("sysroot").map(PathBuf::from));

        let libs = matches
            .opt_strs("L")
            .iter()
            .map(|s| {
                SearchPath::from_cli_opt(
                    sysroot.path(),
                    &target,
                    early_dcx,
                    s,
                    #[allow(rustc::bad_opt_access)] // we have no `Session` here
                    unstable_opts.unstable_options,
                )
            })
            .collect();

        let crate_types = match parse_crate_types_from_list(matches.opt_strs("crate-type")) {
            Ok(types) => types,
            Err(e) => {
                dcx.fatal(format!("unknown crate type: {e}"));
            }
        };

        let bin_crate = crate_types.contains(&CrateType::Executable);
        let proc_macro_crate = crate_types.contains(&CrateType::ProcMacro);
        let playground_url = matches.opt_str("playground-url");
        let module_sorting = if matches.opt_present("sort-modules-by-appearance") {
            ModuleSorting::DeclarationOrder
        } else {
            ModuleSorting::Alphabetical
        };
        let resource_suffix = matches.opt_str("resource-suffix").unwrap_or_default();
        let markdown_no_toc = matches.opt_present("markdown-no-toc");
        let markdown_css = matches.opt_strs("markdown-css");
        let markdown_playground_url = matches.opt_str("markdown-playground-url");
        let crate_version = matches.opt_str("crate-version");
        let enable_index_page = matches.opt_present("enable-index-page") || index_page.is_some();
        let static_root_path = matches.opt_str("static-root-path");
        let test_run_directory = matches.opt_str("test-run-directory").map(PathBuf::from);
        let persist_doctests = matches.opt_str("persist-doctests").map(PathBuf::from);
        let test_builder = matches.opt_str("test-builder").map(PathBuf::from);
        let codegen_options_strs = matches.opt_strs("C");
        let unstable_opts_strs = matches.opt_strs("Z");
        let lib_strs = matches.opt_strs("L");
        let extern_strs = matches.opt_strs("extern");
        let test_runtool = matches.opt_str("test-runtool");
        let test_runtool_args = matches.opt_strs("test-runtool-arg");
        let document_private = matches.opt_present("document-private-items");
        let document_hidden = matches.opt_present("document-hidden-items");
        let run_check = matches.opt_present("check");
        let generate_redirect_map = matches.opt_present("generate-redirect-map");
        let show_type_layout = matches.opt_present("show-type-layout");
        let no_capture = matches.opt_present("no-capture");
        let generate_link_to_definition = matches.opt_present("generate-link-to-definition");
        let generate_macro_expansion = matches.opt_present("generate-macro-expansion");
        let extern_html_root_takes_precedence =
            matches.opt_present("extern-html-root-takes-precedence");
        let html_no_source = matches.opt_present("html-no-source");
        let should_merge = match parse_merge(matches) {
            Ok(result) => result,
            Err(e) => dcx.fatal(format!("--merge option error: {e}")),
        };

        if generate_link_to_definition && (show_coverage || output_format != OutputFormat::Html) {
            dcx.struct_warn(
                "`--generate-link-to-definition` option can only be used with HTML output format",
            )
            .with_note("`--generate-link-to-definition` option will be ignored")
            .emit();
        }
        if generate_macro_expansion && (show_coverage || output_format != OutputFormat::Html) {
            dcx.struct_warn(
                "`--generate-macro-expansion` option can only be used with HTML output format",
            )
            .with_note("`--generate-macro-expansion` option will be ignored")
            .emit();
        }

        let scrape_examples_options = ScrapeExamplesOptions::new(matches, dcx);
        let with_examples = matches.opt_strs("with-examples");
        let call_locations =
            crate::scrape_examples::load_call_locations(with_examples, dcx, &mut loaded_paths);
        let doctest_build_args = matches.opt_strs("doctest-build-arg");

        let disable_minification = matches.opt_present("disable-minification");

        let options = Options {
            bin_crate,
            proc_macro_crate,
            error_format,
            diagnostic_width,
            libs,
            lib_strs,
            externs,
            extern_strs,
            cfgs,
            check_cfgs,
            codegen_options,
            codegen_options_strs,
            unstable_opts,
            unstable_opts_strs,
            target,
            edition,
            sysroot,
            lint_opts,
            describe_lints,
            lint_cap,
            should_test,
            test_args,
            show_coverage,
            crate_version,
            test_run_directory,
            persist_doctests,
            test_runtool,
            test_runtool_args,
            test_builder,
            run_check,
            no_run,
            test_builder_wrappers,
            remap_path_prefix,
            no_capture,
            crate_name,
            output_format,
            json_unused_externs,
            scrape_examples_options,
            unstable_features,
            doctest_build_args,
            target_modifiers,
        };
        let render_options = RenderOptions {
            output,
            external_html,
            id_map,
            playground_url,
            module_sorting,
            themes,
            extension_css,
            extern_html_root_urls,
            extern_html_root_takes_precedence,
            default_settings,
            resource_suffix,
            enable_index_page,
            index_page,
            static_root_path,
            markdown_no_toc,
            markdown_css,
            markdown_playground_url,
            document_private,
            document_hidden,
            generate_redirect_map,
            show_type_layout,
            unstable_features,
            emit,
            generate_link_to_definition,
            generate_macro_expansion,
            call_locations,
            no_emit_shared: false,
            html_no_source,
            output_to_stdout,
            should_merge,
            include_parts_dir,
            parts_out_dir,
            disable_minification,
        };
        Some((input, options, render_options, loaded_paths))
    }
}

/// Returns `true` if the file given as `self.input` is a Markdown file.
pub(crate) fn markdown_input(input: &Input) -> Option<&Path> {
    input.opt_path().filter(|p| matches!(p.extension(), Some(e) if e == "md" || e == "markdown"))
}

fn parse_remap_path_prefix(
    matches: &getopts::Matches,
) -> Result<Vec<(PathBuf, PathBuf)>, &'static str> {
    matches
        .opt_strs("remap-path-prefix")
        .into_iter()
        .map(|remap| {
            remap
                .rsplit_once('=')
                .ok_or("--remap-path-prefix must contain '=' between FROM and TO")
                .map(|(from, to)| (PathBuf::from(from), PathBuf::from(to)))
        })
        .collect()
}

/// Prints deprecation warnings for deprecated options
fn check_deprecated_options(matches: &getopts::Matches, dcx: DiagCtxtHandle<'_>) {
    let deprecated_flags = [];

    for &flag in deprecated_flags.iter() {
        if matches.opt_present(flag) {
            dcx.struct_warn(format!("the `{flag}` flag is deprecated"))
                .with_note(
                    "see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                    for more information",
                )
                .emit();
        }
    }

    let removed_flags = ["plugins", "plugin-path", "no-defaults", "passes", "input-format"];

    for &flag in removed_flags.iter() {
        if matches.opt_present(flag) {
            let mut err = dcx.struct_warn(format!("the `{flag}` flag no longer functions"));
            err.note(
                "see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                for more information",
            );

            if flag == "no-defaults" || flag == "passes" {
                err.help("you may want to use --document-private-items");
            } else if flag == "plugins" || flag == "plugin-path" {
                err.warn("see CVE-2018-1000622");
            }

            err.emit();
        }
    }
}

/// Extracts `--extern-html-root-url` arguments from `matches` and returns a map of crate names to
/// the given URLs. If an `--extern-html-root-url` argument was ill-formed, returns an error
/// describing the issue.
fn parse_extern_html_roots(
    matches: &getopts::Matches,
) -> Result<BTreeMap<String, String>, &'static str> {
    let mut externs = BTreeMap::new();
    for arg in &matches.opt_strs("extern-html-root-url") {
        let (name, url) =
            arg.split_once('=').ok_or("--extern-html-root-url must be of the form name=url")?;
        externs.insert(name.to_string(), url.to_string());
    }
    Ok(externs)
}

/// Path directly to crate-info file.
///
/// For example, `/home/user/project/target/doc.parts/<crate>/crate-info`.
#[derive(Clone, Debug)]
pub(crate) struct PathToParts(pub(crate) PathBuf);

impl PathToParts {
    fn from_flag(path: String) -> Result<PathToParts, String> {
        let mut path = PathBuf::from(path);
        // check here is for diagnostics
        if path.exists() && !path.is_dir() {
            Err(format!(
                "--parts-out-dir and --include-parts-dir expect directories, found: {}",
                path.display(),
            ))
        } else {
            // if it doesn't exist, we'll create it. worry about that in write_shared
            path.push("crate-info");
            Ok(PathToParts(path))
        }
    }
}

/// Reports error if --include-parts-dir / crate-info is not a file
fn parse_include_parts_dir(m: &getopts::Matches) -> Result<Vec<PathToParts>, String> {
    let mut ret = Vec::new();
    for p in m.opt_strs("include-parts-dir") {
        let p = PathToParts::from_flag(p)?;
        // this is just for diagnostic
        if !p.0.is_file() {
            return Err(format!("--include-parts-dir expected {} to be a file", p.0.display()));
        }
        ret.push(p);
    }
    Ok(ret)
}

/// Controls merging of cross-crate information
#[derive(Debug, Clone)]
pub(crate) struct ShouldMerge {
    /// Should we append to existing cci in the doc root
    pub(crate) read_rendered_cci: bool,
    /// Should we write cci to the doc root
    pub(crate) write_rendered_cci: bool,
}

/// Extracts read_rendered_cci and write_rendered_cci from command line arguments, or
/// reports an error if an invalid option was provided
fn parse_merge(m: &getopts::Matches) -> Result<ShouldMerge, &'static str> {
    match m.opt_str("merge").as_deref() {
        // default = read-write
        None => Ok(ShouldMerge { read_rendered_cci: true, write_rendered_cci: true }),
        Some("none") if m.opt_present("include-parts-dir") => {
            Err("--include-parts-dir not allowed if --merge=none")
        }
        Some("none") => Ok(ShouldMerge { read_rendered_cci: false, write_rendered_cci: false }),
        Some("shared") if m.opt_present("parts-out-dir") || m.opt_present("include-parts-dir") => {
            Err("--parts-out-dir and --include-parts-dir not allowed if --merge=shared")
        }
        Some("shared") => Ok(ShouldMerge { read_rendered_cci: true, write_rendered_cci: true }),
        Some("finalize") if m.opt_present("parts-out-dir") => {
            Err("--parts-out-dir not allowed if --merge=finalize")
        }
        Some("finalize") => Ok(ShouldMerge { read_rendered_cci: false, write_rendered_cci: true }),
        Some(_) => Err("argument to --merge must be `none`, `shared`, or `finalize`"),
    }
}
