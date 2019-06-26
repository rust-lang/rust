use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use errors;
use errors::emitter::{ColorConfig, HumanReadableErrorType};
use getopts;
use rustc::lint::Level;
use rustc::session::early_error;
use rustc::session::config::{CodegenOptions, DebuggingOptions, ErrorOutputType, Externs};
use rustc::session::config::{nightly_options, build_codegen_options, build_debugging_options,
                             get_cmd_lint_options, ExternEntry};
use rustc::session::search_paths::SearchPath;
use rustc_driver;
use rustc_target::spec::TargetTriple;
use syntax::edition::{Edition, DEFAULT_EDITION};

use crate::core::new_handler;
use crate::externalfiles::ExternalHtml;
use crate::html;
use crate::html::{static_files};
use crate::html::markdown::{IdMap};
use crate::opts;
use crate::passes::{self, DefaultPassOption};
use crate::theme;

/// Configuration options for rustdoc.
#[derive(Clone)]
pub struct Options {
    // Basic options / Options passed directly to rustc

    /// The crate root or Markdown file to load.
    pub input: PathBuf,
    /// The name of the crate being documented.
    pub crate_name: Option<String>,
    /// How to format errors and warnings.
    pub error_format: ErrorOutputType,
    /// Library search paths to hand to the compiler.
    pub libs: Vec<SearchPath>,
    /// The list of external crates to link against.
    pub externs: Externs,
    /// List of `cfg` flags to hand to the compiler. Always includes `rustdoc`.
    pub cfgs: Vec<String>,
    /// Codegen options to hand to the compiler.
    pub codegen_options: CodegenOptions,
    /// Debugging (`-Z`) options to pass to the compiler.
    pub debugging_options: DebuggingOptions,
    /// The target used to compile the crate against.
    pub target: Option<TargetTriple>,
    /// Edition used when reading the crate. Defaults to "2015". Also used by default when
    /// compiling doctests from the crate.
    pub edition: Edition,
    /// The path to the sysroot. Used during the compilation process.
    pub maybe_sysroot: Option<PathBuf>,
    /// Linker to use when building doctests.
    pub linker: Option<PathBuf>,
    /// Lint information passed over the command-line.
    pub lint_opts: Vec<(String, Level)>,
    /// Whether to ask rustc to describe the lints it knows. Practically speaking, this will not be
    /// used, since we abort if we have no input file, but it's included for completeness.
    pub describe_lints: bool,
    /// What level to cap lints at.
    pub lint_cap: Option<Level>,

    // Options specific to running doctests

    /// Whether we should run doctests instead of generating docs.
    pub should_test: bool,
    /// List of arguments to pass to the test harness, if running tests.
    pub test_args: Vec<String>,
    /// Optional path to persist the doctest executables to, defaults to a
    /// temporary directory if not set.
    pub persist_doctests: Option<PathBuf>,

    // Options that affect the documentation process

    /// The selected default set of passes to use.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    pub default_passes: DefaultPassOption,
    /// Any passes manually selected by the user.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    pub manual_passes: Vec<String>,
    /// Whether to display warnings during doc generation or while gathering doctests. By default,
    /// all non-rustdoc-specific lints are allowed when generating docs.
    pub display_warnings: bool,
    /// Whether to run the `calculate-doc-coverage` pass, which counts the number of public items
    /// with and without documentation.
    pub show_coverage: bool,

    // Options that alter generated documentation pages

    /// Crate version to note on the sidebar of generated docs.
    pub crate_version: Option<String>,
    /// Collected options specific to outputting final pages.
    pub render_options: RenderOptions,
}

impl fmt::Debug for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct FmtExterns<'a>(&'a Externs);

        impl<'a> fmt::Debug for FmtExterns<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_map()
                    .entries(self.0.iter())
                    .finish()
            }
        }

        f.debug_struct("Options")
            .field("input", &self.input)
            .field("crate_name", &self.crate_name)
            .field("error_format", &self.error_format)
            .field("libs", &self.libs)
            .field("externs", &FmtExterns(&self.externs))
            .field("cfgs", &self.cfgs)
            .field("codegen_options", &"...")
            .field("debugging_options", &"...")
            .field("target", &self.target)
            .field("edition", &self.edition)
            .field("maybe_sysroot", &self.maybe_sysroot)
            .field("linker", &self.linker)
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
            .finish()
    }
}

/// Configuration options for the HTML page-creation process.
#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// Output directory to generate docs into. Defaults to `doc`.
    pub output: PathBuf,
    /// External files to insert into generated pages.
    pub external_html: ExternalHtml,
    /// A pre-populated `IdMap` with the default headings and any headings added by Markdown files
    /// processed by `external_html`.
    pub id_map: IdMap,
    /// If present, playground URL to use in the "Run" button added to code samples.
    ///
    /// Be aware: This option can come both from the CLI and from crate attributes!
    pub playground_url: Option<String>,
    /// Whether to sort modules alphabetically on a module page instead of using declaration order.
    /// `true` by default.
    //
    // FIXME(misdreavus): the flag name is `--sort-modules-by-appearance` but the meaning is
    // inverted once read.
    pub sort_modules_alphabetically: bool,
    /// List of themes to extend the docs with. Original argument name is included to assist in
    /// displaying errors if it fails a theme check.
    pub themes: Vec<PathBuf>,
    /// If present, CSS file that contains rules to add to the default CSS.
    pub extension_css: Option<PathBuf>,
    /// A map of crate names to the URL to use instead of querying the crate's `html_root_url`.
    pub extern_html_root_urls: BTreeMap<String, String>,
    /// If present, suffix added to CSS/JavaScript files when referencing them in generated pages.
    pub resource_suffix: String,
    /// Whether to run the static CSS/JavaScript through a minifier when outputting them. `true` by
    /// default.
    //
    // FIXME(misdreavus): the flag name is `--disable-minification` but the meaning is inverted
    // once read.
    pub enable_minification: bool,
    /// Whether to create an index page in the root of the output directory. If this is true but
    /// `enable_index_page` is None, generate a static listing of crates instead.
    pub enable_index_page: bool,
    /// A file to use as the index page at the root of the output directory. Overrides
    /// `enable_index_page` to be true if set.
    pub index_page: Option<PathBuf>,
    /// An optional path to use as the location of static files. If not set, uses combinations of
    /// `../` to reach the documentation root.
    pub static_root_path: Option<String>,

    // Options specific to reading standalone Markdown files

    /// Whether to generate a table of contents on the output file when reading a standalone
    /// Markdown file.
    pub markdown_no_toc: bool,
    /// Additional CSS files to link in pages generated from standalone Markdown files.
    pub markdown_css: Vec<String>,
    /// If present, playground URL to use in the "Run" button added to code samples generated from
    /// standalone Markdown files. If not present, `playground_url` is used.
    pub markdown_playground_url: Option<String>,
    /// If false, the `select` element to have search filtering by crates on rendered docs
    /// won't be generated.
    pub generate_search_filter: bool,
    /// Option (disabled by default) to generate files used by RLS and some other tools.
    pub generate_redirect_pages: bool,
}

impl Options {
    /// Parses the given command-line for options. If an error message or other early-return has
    /// been printed, returns `Err` with the exit code.
    pub fn from_matches(matches: &getopts::Matches) -> Result<Options, i32> {
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
            for &name in passes::DEFAULT_PASSES {
                println!("{:>20}", name);
            }
            println!("\nPasses run with `--document-private-items`:");
            for &name in passes::DEFAULT_PRIVATE_PASSES {
                println!("{:>20}", name);
            }

            if nightly_options::is_nightly_build() {
                println!("\nPasses run with `--show-coverage`:");
                for &name in passes::DEFAULT_COVERAGE_PASSES {
                    println!("{:>20}", name);
                }
                println!("\nPasses run with `--show-coverage --document-private-items`:");
                for &name in passes::PRIVATE_COVERAGE_PASSES {
                    println!("{:>20}", name);
                }
            }

            return Err(0);
        }

        let color = match matches.opt_str("color").as_ref().map(|s| &s[..]) {
            Some("auto") => ColorConfig::Auto,
            Some("always") => ColorConfig::Always,
            Some("never") => ColorConfig::Never,
            None => ColorConfig::Auto,
            Some(arg) => {
                early_error(ErrorOutputType::default(),
                            &format!("argument for --color must be `auto`, `always` or `never` \
                                      (instead was `{}`)", arg));
            }
        };
        // FIXME: deduplicate this code from the identical code in librustc/session/config.rs
        let error_format = match matches.opt_str("error-format").as_ref().map(|s| &s[..]) {
            None |
            Some("human") => ErrorOutputType::HumanReadable(HumanReadableErrorType::Default(color)),
            Some("json") => ErrorOutputType::Json {
                pretty: false,
                json_rendered: HumanReadableErrorType::Default(ColorConfig::Never),
            },
            Some("pretty-json") => ErrorOutputType::Json {
                pretty: true,
                json_rendered: HumanReadableErrorType::Default(ColorConfig::Never),
            },
            Some("short") => ErrorOutputType::HumanReadable(HumanReadableErrorType::Short(color)),
            Some(arg) => {
                early_error(ErrorOutputType::default(),
                            &format!("argument for --error-format must be `human`, `json` or \
                                      `short` (instead was `{}`)", arg));
            }
        };

        let codegen_options = build_codegen_options(matches, error_format);
        let debugging_options = build_debugging_options(matches, error_format);

        let diag = new_handler(error_format,
                               None,
                               debugging_options.treat_err_as_bug,
                               debugging_options.ui_testing);

        // check for deprecated options
        check_deprecated_options(&matches, &diag);

        let to_check = matches.opt_strs("theme-checker");
        if !to_check.is_empty() {
            let paths = theme::load_css_paths(static_files::themes::LIGHT.as_bytes());
            let mut errors = 0;

            println!("rustdoc: [theme-checker] Starting tests!");
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

        let libs = matches.opt_strs("L").iter()
            .map(|s| SearchPath::from_cli_opt(s, error_format))
            .collect();
        let externs = match parse_externs(&matches) {
            Ok(ex) => ex,
            Err(err) => {
                diag.struct_err(&err).emit();
                return Err(1);
            }
        };
        let extern_html_root_urls = match parse_extern_html_roots(&matches) {
            Ok(ex) => ex,
            Err(err) => {
                diag.struct_err(err).emit();
                return Err(1);
            }
        };

        let test_args = matches.opt_strs("test-args");
        let test_args: Vec<String> = test_args.iter()
                                              .flat_map(|s| s.split_whitespace())
                                              .map(|s| s.to_string())
                                              .collect();

        let should_test = matches.opt_present("test");

        let output = matches.opt_str("o")
                            .map(|s| PathBuf::from(&s))
                            .unwrap_or_else(|| PathBuf::from("doc"));
        let mut cfgs = matches.opt_strs("cfg");
        cfgs.push("rustdoc".to_string());

        let extension_css = matches.opt_str("e").map(|s| PathBuf::from(&s));

        if let Some(ref p) = extension_css {
            if !p.is_file() {
                diag.struct_err("option --extend-css argument must be a file").emit();
                return Err(1);
            }
        }

        let mut themes = Vec::new();
        if matches.opt_present("themes") {
            let paths = theme::load_css_paths(static_files::themes::LIGHT.as_bytes());

            for (theme_file, theme_s) in matches.opt_strs("themes")
                                                .iter()
                                                .map(|s| (PathBuf::from(&s), s.to_owned())) {
                if !theme_file.is_file() {
                    diag.struct_err("option --themes arguments must all be files").emit();
                    return Err(1);
                }
                let (success, ret) = theme::test_theme_against(&theme_file, &paths, &diag);
                if !success || !ret.is_empty() {
                    diag.struct_err(&format!("invalid theme: \"{}\"", theme_s))
                        .help("check what's wrong with the --theme-checker option")
                        .emit();
                    return Err(1);
                }
                themes.push(theme_file);
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
                &diag, &mut id_map, edition) {
            Some(eh) => eh,
            None => return Err(3),
        };

        match matches.opt_str("r").as_ref().map(|s| &**s) {
            Some("rust") | None => {}
            Some(s) => {
                diag.struct_err(&format!("unknown input format: {}", s)).emit();
                return Err(1);
            }
        }

        match matches.opt_str("w").as_ref().map(|s| &**s) {
            Some("html") | None => {}
            Some(s) => {
                diag.struct_err(&format!("unknown output format: {}", s)).emit();
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

        let target = matches.opt_str("target").map(|target| {
            if target.ends_with(".json") {
                TargetTriple::TargetPath(PathBuf::from(target))
            } else {
                TargetTriple::TargetTriple(target)
            }
        });

        let show_coverage = matches.opt_present("show-coverage");
        let document_private = matches.opt_present("document-private-items");

        let default_passes = if matches.opt_present("no-defaults") {
            passes::DefaultPassOption::None
        } else if show_coverage && document_private {
            passes::DefaultPassOption::PrivateCoverage
        } else if show_coverage {
            passes::DefaultPassOption::Coverage
        } else if document_private {
            passes::DefaultPassOption::Private
        } else {
            passes::DefaultPassOption::Default
        };
        let manual_passes = matches.opt_strs("passes");

        let crate_name = matches.opt_str("crate-name");
        let playground_url = matches.opt_str("playground-url");
        let maybe_sysroot = matches.opt_str("sysroot").map(PathBuf::from);
        let display_warnings = matches.opt_present("display-warnings");
        let linker = matches.opt_str("linker").map(PathBuf::from);
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
        let generate_redirect_pages = matches.opt_present("generate-redirect-pages");

        let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(matches, error_format);

        Ok(Options {
            input,
            crate_name,
            error_format,
            libs,
            externs,
            cfgs,
            codegen_options,
            debugging_options,
            target,
            edition,
            maybe_sysroot,
            linker,
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
            render_options: RenderOptions {
                output,
                external_html,
                id_map,
                playground_url,
                sort_modules_alphabetically,
                themes,
                extension_css,
                extern_html_root_urls,
                resource_suffix,
                enable_minification,
                enable_index_page,
                index_page,
                static_root_path,
                markdown_no_toc,
                markdown_css,
                markdown_playground_url,
                generate_search_filter,
                generate_redirect_pages,
            }
        })
    }

    /// Returns `true` if the file given as `self.input` is a Markdown file.
    pub fn markdown_input(&self) -> bool {
        self.input.extension()
            .map_or(false, |e| e == "md" || e == "markdown")
    }
}

/// Prints deprecation warnings for deprecated options
fn check_deprecated_options(matches: &getopts::Matches, diag: &errors::Handler) {
    let deprecated_flags = [
       "input-format",
       "output-format",
       "no-defaults",
       "passes",
    ];

    for flag in deprecated_flags.iter() {
        if matches.opt_present(flag) {
            let mut err = diag.struct_warn(&format!("the '{}' flag is considered deprecated",
                                                    flag));
            err.warn("please see https://github.com/rust-lang/rust/issues/44136");

            if *flag == "no-defaults" {
                err.help("you may want to use --document-private-items");
            }

            err.emit();
        }
    }

    let removed_flags = [
        "plugins",
        "plugin-path",
    ];

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
        let mut parts = arg.splitn(2, '=');
        let name = parts.next().ok_or("--extern-html-root-url must not be empty")?;
        let url = parts.next().ok_or("--extern-html-root-url must be of the form name=url")?;
        externs.insert(name.to_string(), url.to_string());
    }

    Ok(externs)
}

/// Extracts `--extern CRATE=PATH` arguments from `matches` and
/// returns a map mapping crate names to their paths or else an
/// error message.
// FIXME(eddyb) This shouldn't be duplicated with `rustc::session`.
fn parse_externs(matches: &getopts::Matches) -> Result<Externs, String> {
    let mut externs: BTreeMap<_, ExternEntry> = BTreeMap::new();
    for arg in &matches.opt_strs("extern") {
        let mut parts = arg.splitn(2, '=');
        let name = parts.next().ok_or("--extern value must not be empty".to_string())?;
        let location = parts.next().map(|s| s.to_string());
        if location.is_none() && !nightly_options::is_unstable_enabled(matches) {
            return Err("the `-Z unstable-options` flag must also be passed to \
                        enable `--extern crate_name` without `=path`".to_string());
        }
        let name = name.to_string();
        // For Rustdoc purposes, we can treat all externs as public
        externs.entry(name)
            .or_default()
            .locations.insert(location.clone());
    }
    Ok(Externs::new(externs))
}
