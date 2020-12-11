use std::collections::{BTreeMap, HashMap};
use std::convert::TryFrom;
use std::ffi::OsStr;
use std::fmt;
use std::path::PathBuf;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::DefId;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_session::config::{self, parse_crate_types_from_list, parse_externs, CrateType};
use rustc_session::config::{
    build_codegen_options, build_debugging_options, get_cmd_lint_options, host_triple,
    nightly_options,
};
use rustc_session::config::{CodegenOptions, DebuggingOptions, ErrorOutputType, Externs};
use rustc_session::getopts;
use rustc_session::lint::Level;
use rustc_session::search_paths::SearchPath;
use rustc_span::edition::{Edition, DEFAULT_EDITION};
use rustc_target::spec::TargetTriple;

use crate::core::new_handler;
use crate::externalfiles::ExternalHtml;
use crate::html;
use crate::html::markdown::IdMap;
use crate::html::render::StylePath;
use crate::html::static_files;
use crate::opts;
use crate::passes::{self, Condition, DefaultPassOption};
use crate::theme;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
crate enum OutputFormat {
    Json,
    Html,
}

impl OutputFormat {
    crate fn is_json(&self) -> bool {
        match self {
            OutputFormat::Json => true,
            _ => false,
        }
    }
}

impl TryFrom<&str> for OutputFormat {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "json" => Ok(OutputFormat::Json),
            "html" => Ok(OutputFormat::Html),
            _ => Err(format!("unknown output format `{}`", value)),
        }
    }
}

/// Configuration options for rustdoc.
#[derive(Clone)]
crate struct Options {
    // Basic options / Options passed directly to rustc
    /// The crate root or Markdown file to load.
    crate input: PathBuf,
    /// The name of the crate being documented.
    crate crate_name: Option<String>,
    /// Whether or not this is a proc-macro crate
    crate proc_macro_crate: bool,
    /// How to format errors and warnings.
    crate error_format: ErrorOutputType,
    /// Library search paths to hand to the compiler.
    crate libs: Vec<SearchPath>,
    /// Library search paths strings to hand to the compiler.
    crate lib_strs: Vec<String>,
    /// The list of external crates to link against.
    crate externs: Externs,
    /// The list of external crates strings to link against.
    crate extern_strs: Vec<String>,
    /// List of `cfg` flags to hand to the compiler. Always includes `rustdoc`.
    crate cfgs: Vec<String>,
    /// Codegen options to hand to the compiler.
    crate codegen_options: CodegenOptions,
    /// Codegen options strings to hand to the compiler.
    crate codegen_options_strs: Vec<String>,
    /// Debugging (`-Z`) options to pass to the compiler.
    crate debugging_opts: DebuggingOptions,
    /// Debugging (`-Z`) options strings to pass to the compiler.
    crate debugging_opts_strs: Vec<String>,
    /// The target used to compile the crate against.
    crate target: TargetTriple,
    /// Edition used when reading the crate. Defaults to "2015". Also used by default when
    /// compiling doctests from the crate.
    crate edition: Edition,
    /// The path to the sysroot. Used during the compilation process.
    crate maybe_sysroot: Option<PathBuf>,
    /// Lint information passed over the command-line.
    crate lint_opts: Vec<(String, Level)>,
    /// Whether to ask rustc to describe the lints it knows. Practically speaking, this will not be
    /// used, since we abort if we have no input file, but it's included for completeness.
    crate describe_lints: bool,
    /// What level to cap lints at.
    crate lint_cap: Option<Level>,

    // Options specific to running doctests
    /// Whether we should run doctests instead of generating docs.
    crate should_test: bool,
    /// List of arguments to pass to the test harness, if running tests.
    crate test_args: Vec<String>,
    /// Optional path to persist the doctest executables to, defaults to a
    /// temporary directory if not set.
    crate persist_doctests: Option<PathBuf>,
    /// Runtool to run doctests with
    crate runtool: Option<String>,
    /// Arguments to pass to the runtool
    crate runtool_args: Vec<String>,
    /// Whether to allow ignoring doctests on a per-target basis
    /// For example, using ignore-foo to ignore running the doctest on any target that
    /// contains "foo" as a substring
    crate enable_per_target_ignores: bool,

    /// The path to a rustc-like binary to build tests with. If not set, we
    /// default to loading from $sysroot/bin/rustc.
    crate test_builder: Option<PathBuf>,

    // Options that affect the documentation process
    /// The selected default set of passes to use.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    crate default_passes: DefaultPassOption,
    /// Any passes manually selected by the user.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    crate manual_passes: Vec<String>,
    /// Whether to display warnings during doc generation or while gathering doctests. By default,
    /// all non-rustdoc-specific lints are allowed when generating docs.
    crate display_warnings: bool,
    /// Whether to run the `calculate-doc-coverage` pass, which counts the number of public items
    /// with and without documentation.
    crate show_coverage: bool,

    // Options that alter generated documentation pages
    /// Crate version to note on the sidebar of generated docs.
    crate crate_version: Option<String>,
    /// Collected options specific to outputting final pages.
    crate render_options: RenderOptions,
    /// Output format rendering (used only for "show-coverage" option for the moment)
    crate output_format: Option<OutputFormat>,
    /// If this option is set to `true`, rustdoc will only run checks and not generate
    /// documentation.
    crate run_check: bool,
}

impl fmt::Debug for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct FmtExterns<'a>(&'a Externs);

        impl<'a> fmt::Debug for FmtExterns<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_map().entries(self.0.iter()).finish()
            }
        }

        f.debug_struct("Options")
            .field("input", &self.input)
            .field("crate_name", &self.crate_name)
            .field("proc_macro_crate", &self.proc_macro_crate)
            .field("error_format", &self.error_format)
            .field("libs", &self.libs)
            .field("externs", &FmtExterns(&self.externs))
            .field("cfgs", &self.cfgs)
            .field("codegen_options", &"...")
            .field("debugging_options", &"...")
            .field("target", &self.target)
            .field("edition", &self.edition)
            .field("maybe_sysroot", &self.maybe_sysroot)
            .field("lint_opts", &self.lint_opts)
            .field("describe_lints", &self.describe_lints)
            .field("lint_cap", &self.lint_cap)
            .field("should_test", &self.should_test)
            .field("test_args", &self.test_args)
            .field("persist_doctests", &self.persist_doctests)
            .field("default_passes", &self.default_passes)
            .field("manual_passes", &self.manual_passes)
            .field("display_warnings", &self.display_warnings)
            .field("show_coverage", &self.show_coverage)
            .field("crate_version", &self.crate_version)
            .field("render_options", &self.render_options)
            .field("runtool", &self.runtool)
            .field("runtool_args", &self.runtool_args)
            .field("enable-per-target-ignores", &self.enable_per_target_ignores)
            .field("run_check", &self.run_check)
            .finish()
    }
}

/// Configuration options for the HTML page-creation process.
#[derive(Clone, Debug)]
crate struct RenderOptions {
    /// Output directory to generate docs into. Defaults to `doc`.
    crate output: PathBuf,
    /// External files to insert into generated pages.
    crate external_html: ExternalHtml,
    /// A pre-populated `IdMap` with the default headings and any headings added by Markdown files
    /// processed by `external_html`.
    crate id_map: IdMap,
    /// If present, playground URL to use in the "Run" button added to code samples.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    crate playground_url: Option<String>,
    /// Whether to sort modules alphabetically on a module page instead of using declaration order.
    /// `true` by default.
    //
    // FIXME(misdreavus): the flag name is `--sort-modules-by-appearance` but the meaning is
    // inverted once read.
    crate sort_modules_alphabetically: bool,
    /// List of themes to extend the docs with. Original argument name is included to assist in
    /// displaying errors if it fails a theme check.
    crate themes: Vec<StylePath>,
    /// If present, CSS file that contains rules to add to the default CSS.
    crate extension_css: Option<PathBuf>,
    /// A map of crate names to the URL to use instead of querying the crate's `html_root_url`.
    crate extern_html_root_urls: BTreeMap<String, String>,
    /// A map of the default settings (values are as for DOM storage API). Keys should lack the
    /// `rustdoc-` prefix.
    crate default_settings: HashMap<String, String>,
    /// If present, suffix added to CSS/JavaScript files when referencing them in generated pages.
    crate resource_suffix: String,
    /// Whether to run the static CSS/JavaScript through a minifier when outputting them. `true` by
    /// default.
    //
    // FIXME(misdreavus): the flag name is `--disable-minification` but the meaning is inverted
    // once read.
    crate enable_minification: bool,
    /// Whether to create an index page in the root of the output directory. If this is true but
    /// `enable_index_page` is None, generate a static listing of crates instead.
    crate enable_index_page: bool,
    /// A file to use as the index page at the root of the output directory. Overrides
    /// `enable_index_page` to be true if set.
    crate index_page: Option<PathBuf>,
    /// An optional path to use as the location of static files. If not set, uses combinations of
    /// `../` to reach the documentation root.
    crate static_root_path: Option<String>,

    // Options specific to reading standalone Markdown files
    /// Whether to generate a table of contents on the output file when reading a standalone
    /// Markdown file.
    crate markdown_no_toc: bool,
    /// Additional CSS files to link in pages generated from standalone Markdown files.
    crate markdown_css: Vec<String>,
    /// If present, playground URL to use in the "Run" button added to code samples generated from
    /// standalone Markdown files. If not present, `playground_url` is used.
    crate markdown_playground_url: Option<String>,
    /// If false, the `select` element to have search filtering by crates on rendered docs
    /// won't be generated.
    crate generate_search_filter: bool,
    /// Document items that have lower than `pub` visibility.
    crate document_private: bool,
    /// Document items that have `doc(hidden)`.
    crate document_hidden: bool,
    crate unstable_features: rustc_feature::UnstableFeatures,
}

/// Temporary storage for data obtained during `RustdocVisitor::clean()`.
/// Later on moved into `CACHE_KEY`.
#[derive(Default, Clone)]
crate struct RenderInfo {
    crate inlined: FxHashSet<DefId>,
    crate external_paths: crate::core::ExternalPaths,
    crate exact_paths: FxHashMap<DefId, Vec<String>>,
    crate access_levels: AccessLevels<DefId>,
    crate deref_trait_did: Option<DefId>,
    crate deref_mut_trait_did: Option<DefId>,
    crate owned_box_did: Option<DefId>,
    crate output_format: Option<OutputFormat>,
}

impl Options {
    /// Parses the given command-line for options. If an error message or other early-return has
    /// been printed, returns `Err` with the exit code.
    crate fn from_matches(matches: &getopts::Matches) -> Result<Options, i32> {
        // Check for unstable options.
        nightly_options::check_nightly_options(&matches, &opts());

        if matches.opt_present("h") || matches.opt_present("help") {
            crate::usage("rustdoc");
            return Err(0);
        } else if matches.opt_present("version") {
            rustc_driver::version("rustdoc", &matches);
            return Err(0);
        }

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

            return Err(0);
        }

        let color = config::parse_color(&matches);
        let (json_rendered, _artifacts) = config::parse_json(&matches);
        let error_format = config::parse_error_format(&matches, color, json_rendered);

        let codegen_options = build_codegen_options(matches, error_format);
        let debugging_opts = build_debugging_options(matches, error_format);

        let diag = new_handler(error_format, None, &debugging_opts);

        // check for deprecated options
        check_deprecated_options(&matches, &diag);

        let to_check = matches.opt_strs("check-theme");
        if !to_check.is_empty() {
            let paths = theme::load_css_paths(static_files::themes::LIGHT.as_bytes());
            let mut errors = 0;

            println!("rustdoc: [check-theme] Starting tests! (Ignoring all other arguments)");
            for theme_file in to_check.iter() {
                print!(" - Checking \"{}\"...", theme_file);
                let (success, differences) = theme::test_theme_against(theme_file, &paths, &diag);
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
                return Err(1);
            }
            return Err(0);
        }

        if matches.free.is_empty() {
            diag.struct_err("missing file operand").emit();
            return Err(1);
        }
        if matches.free.len() > 1 {
            diag.struct_err("too many file operands").emit();
            return Err(1);
        }
        let input = PathBuf::from(&matches.free[0]);

        let libs = matches
            .opt_strs("L")
            .iter()
            .map(|s| SearchPath::from_cli_opt(s, error_format))
            .collect();
        let externs = parse_externs(&matches, &debugging_opts, error_format);
        let extern_html_root_urls = match parse_extern_html_roots(&matches) {
            Ok(ex) => ex,
            Err(err) => {
                diag.struct_err(err).emit();
                return Err(1);
            }
        };

        let default_settings: Vec<Vec<(String, String)>> = vec![
            matches
                .opt_str("default-theme")
                .iter()
                .map(|theme| {
                    vec![
                        ("use-system-theme".to_string(), "false".to_string()),
                        ("theme".to_string(), theme.to_string()),
                    ]
                })
                .flatten()
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
        let default_settings = default_settings.into_iter().flatten().collect();

        let test_args = matches.opt_strs("test-args");
        let test_args: Vec<String> =
            test_args.iter().flat_map(|s| s.split_whitespace()).map(|s| s.to_string()).collect();

        let should_test = matches.opt_present("test");

        let output =
            matches.opt_str("o").map(|s| PathBuf::from(&s)).unwrap_or_else(|| PathBuf::from("doc"));
        let cfgs = matches.opt_strs("cfg");

        let extension_css = matches.opt_str("e").map(|s| PathBuf::from(&s));

        if let Some(ref p) = extension_css {
            if !p.is_file() {
                diag.struct_err("option --extend-css argument must be a file").emit();
                return Err(1);
            }
        }

        let mut themes = Vec::new();
        if matches.opt_present("theme") {
            let paths = theme::load_css_paths(static_files::themes::LIGHT.as_bytes());

            for (theme_file, theme_s) in
                matches.opt_strs("theme").iter().map(|s| (PathBuf::from(&s), s.to_owned()))
            {
                if !theme_file.is_file() {
                    diag.struct_err(&format!("invalid argument: \"{}\"", theme_s))
                        .help("arguments to --theme must be files")
                        .emit();
                    return Err(1);
                }
                if theme_file.extension() != Some(OsStr::new("css")) {
                    diag.struct_err(&format!("invalid argument: \"{}\"", theme_s)).emit();
                    return Err(1);
                }
                let (success, ret) = theme::test_theme_against(&theme_file, &paths, &diag);
                if !success {
                    diag.struct_err(&format!("error loading theme file: \"{}\"", theme_s)).emit();
                    return Err(1);
                } else if !ret.is_empty() {
                    diag.struct_warn(&format!(
                        "theme file \"{}\" is missing CSS rules from the default theme",
                        theme_s
                    ))
                    .warn("the theme may appear incorrect when loaded")
                    .help(&format!(
                        "to see what rules are missing, call `rustdoc  --check-theme \"{}\"`",
                        theme_s
                    ))
                    .emit();
                }
                themes.push(StylePath { path: theme_file, disabled: true });
            }
        }

        let edition = if let Some(e) = matches.opt_str("edition") {
            match e.parse() {
                Ok(e) => e,
                Err(_) => {
                    diag.struct_err("could not parse edition").emit();
                    return Err(1);
                }
            }
        } else {
            DEFAULT_EDITION
        };

        let mut id_map = html::markdown::IdMap::new();
        id_map.populate(html::render::initial_ids());
        let external_html = match ExternalHtml::load(
            &matches.opt_strs("html-in-header"),
            &matches.opt_strs("html-before-content"),
            &matches.opt_strs("html-after-content"),
            &matches.opt_strs("markdown-before-content"),
            &matches.opt_strs("markdown-after-content"),
            nightly_options::match_is_nightly_build(&matches),
            &diag,
            &mut id_map,
            edition,
            &None,
        ) {
            Some(eh) => eh,
            None => return Err(3),
        };

        match matches.opt_str("r").as_deref() {
            Some("rust") | None => {}
            Some(s) => {
                diag.struct_err(&format!("unknown input format: {}", s)).emit();
                return Err(1);
            }
        }

        let index_page = matches.opt_str("index-page").map(|s| PathBuf::from(&s));
        if let Some(ref index_page) = index_page {
            if !index_page.is_file() {
                diag.struct_err("option `--index-page` argument must be a file").emit();
                return Err(1);
            }
        }

        let target =
            matches.opt_str("target").map_or(TargetTriple::from_triple(host_triple()), |target| {
                if target.ends_with(".json") {
                    TargetTriple::TargetPath(PathBuf::from(target))
                } else {
                    TargetTriple::TargetTriple(target)
                }
            });

        let show_coverage = matches.opt_present("show-coverage");

        let default_passes = if matches.opt_present("no-defaults") {
            passes::DefaultPassOption::None
        } else if show_coverage {
            passes::DefaultPassOption::Coverage
        } else {
            passes::DefaultPassOption::Default
        };
        let manual_passes = matches.opt_strs("passes");

        let crate_types = match parse_crate_types_from_list(matches.opt_strs("crate-type")) {
            Ok(types) => types,
            Err(e) => {
                diag.struct_err(&format!("unknown crate type: {}", e)).emit();
                return Err(1);
            }
        };

        let output_format = match matches.opt_str("output-format") {
            Some(s) => match OutputFormat::try_from(s.as_str()) {
                Ok(o) => {
                    if o.is_json()
                        && !(show_coverage || nightly_options::match_is_nightly_build(matches))
                    {
                        diag.struct_err("json output format isn't supported for doc generation")
                            .emit();
                        return Err(1);
                    } else if !o.is_json() && show_coverage {
                        diag.struct_err(
                            "html output format isn't supported for the --show-coverage option",
                        )
                        .emit();
                        return Err(1);
                    }
                    Some(o)
                }
                Err(e) => {
                    diag.struct_err(&e).emit();
                    return Err(1);
                }
            },
            None => None,
        };
        let crate_name = matches.opt_str("crate-name");
        let proc_macro_crate = crate_types.contains(&CrateType::ProcMacro);
        let playground_url = matches.opt_str("playground-url");
        let maybe_sysroot = matches.opt_str("sysroot").map(PathBuf::from);
        let display_warnings = matches.opt_present("display-warnings");
        let sort_modules_alphabetically = !matches.opt_present("sort-modules-by-appearance");
        let resource_suffix = matches.opt_str("resource-suffix").unwrap_or_default();
        let enable_minification = !matches.opt_present("disable-minification");
        let markdown_no_toc = matches.opt_present("markdown-no-toc");
        let markdown_css = matches.opt_strs("markdown-css");
        let markdown_playground_url = matches.opt_str("markdown-playground-url");
        let crate_version = matches.opt_str("crate-version");
        let enable_index_page = matches.opt_present("enable-index-page") || index_page.is_some();
        let static_root_path = matches.opt_str("static-root-path");
        let generate_search_filter = !matches.opt_present("disable-per-crate-search");
        let persist_doctests = matches.opt_str("persist-doctests").map(PathBuf::from);
        let test_builder = matches.opt_str("test-builder").map(PathBuf::from);
        let codegen_options_strs = matches.opt_strs("C");
        let debugging_opts_strs = matches.opt_strs("Z");
        let lib_strs = matches.opt_strs("L");
        let extern_strs = matches.opt_strs("extern");
        let runtool = matches.opt_str("runtool");
        let runtool_args = matches.opt_strs("runtool-arg");
        let enable_per_target_ignores = matches.opt_present("enable-per-target-ignores");
        let document_private = matches.opt_present("document-private-items");
        let document_hidden = matches.opt_present("document-hidden-items");
        let run_check = matches.opt_present("check");

        let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(matches, error_format);

        Ok(Options {
            input,
            proc_macro_crate,
            error_format,
            libs,
            lib_strs,
            externs,
            extern_strs,
            cfgs,
            codegen_options,
            codegen_options_strs,
            debugging_opts,
            debugging_opts_strs,
            target,
            edition,
            maybe_sysroot,
            lint_opts,
            describe_lints,
            lint_cap,
            should_test,
            test_args,
            default_passes,
            manual_passes,
            display_warnings,
            show_coverage,
            crate_version,
            persist_doctests,
            runtool,
            runtool_args,
            enable_per_target_ignores,
            test_builder,
            run_check,
            render_options: RenderOptions {
                output,
                external_html,
                id_map,
                playground_url,
                sort_modules_alphabetically,
                themes,
                extension_css,
                extern_html_root_urls,
                default_settings,
                resource_suffix,
                enable_minification,
                enable_index_page,
                index_page,
                static_root_path,
                markdown_no_toc,
                markdown_css,
                markdown_playground_url,
                generate_search_filter,
                document_private,
                document_hidden,
                unstable_features: rustc_feature::UnstableFeatures::from_environment(
                    crate_name.as_deref(),
                ),
            },
            crate_name,
            output_format,
        })
    }

    /// Returns `true` if the file given as `self.input` is a Markdown file.
    crate fn markdown_input(&self) -> bool {
        self.input.extension().map_or(false, |e| e == "md" || e == "markdown")
    }
}

/// Prints deprecation warnings for deprecated options
fn check_deprecated_options(matches: &getopts::Matches, diag: &rustc_errors::Handler) {
    let deprecated_flags = ["input-format", "output-format", "no-defaults", "passes"];

    for flag in deprecated_flags.iter() {
        if matches.opt_present(flag) {
            if *flag == "output-format"
                && (matches.opt_present("show-coverage")
                    || nightly_options::match_is_nightly_build(matches))
            {
                continue;
            }
            let mut err =
                diag.struct_warn(&format!("the '{}' flag is considered deprecated", flag));
            err.warn(
                "see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                 for more information",
            );

            if *flag == "no-defaults" {
                err.help("you may want to use --document-private-items");
            }

            err.emit();
        }
    }

    let removed_flags = ["plugins", "plugin-path"];

    for &flag in removed_flags.iter() {
        if matches.opt_present(flag) {
            diag.struct_warn(&format!("the '{}' flag no longer functions", flag))
                .warn("see CVE-2018-1000622")
                .emit();
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
