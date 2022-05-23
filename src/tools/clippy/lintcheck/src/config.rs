use clap::{App, Arg, ArgMatches};
use std::env;
use std::path::PathBuf;

fn get_clap_config<'a>() -> ArgMatches<'a> {
    App::new("lintcheck")
        .about("run clippy on a set of crates and check output")
        .arg(
            Arg::with_name("only")
                .takes_value(true)
                .value_name("CRATE")
                .long("only")
                .help("Only process a single crate of the list"),
        )
        .arg(
            Arg::with_name("crates-toml")
                .takes_value(true)
                .value_name("CRATES-SOURCES-TOML-PATH")
                .long("crates-toml")
                .help("Set the path for a crates.toml where lintcheck should read the sources from"),
        )
        .arg(
            Arg::with_name("threads")
                .takes_value(true)
                .value_name("N")
                .short("j")
                .long("jobs")
                .help("Number of threads to use, 0 automatic choice"),
        )
        .arg(
            Arg::with_name("fix")
                .long("--fix")
                .help("Runs cargo clippy --fix and checks if all suggestions apply"),
        )
        .arg(
            Arg::with_name("filter")
                .long("--filter")
                .takes_value(true)
                .multiple(true)
                .value_name("clippy_lint_name")
                .help("Apply a filter to only collect specified lints, this also overrides `allow` attributes"),
        )
        .arg(
            Arg::with_name("markdown")
                .long("--markdown")
                .help("Change the reports table to use markdown links"),
        )
        .get_matches()
}

#[derive(Debug)]
pub(crate) struct LintcheckConfig {
    /// max number of jobs to spawn (default 1)
    pub max_jobs: usize,
    /// we read the sources to check from here
    pub sources_toml_path: PathBuf,
    /// we save the clippy lint results here
    pub lintcheck_results_path: PathBuf,
    /// Check only a specified package
    pub only: Option<String>,
    /// whether to just run --fix and not collect all the warnings
    pub fix: bool,
    /// A list of lints that this lintcheck run should focus on
    pub lint_filter: Vec<String>,
    /// Indicate if the output should support markdown syntax
    pub markdown: bool,
}

impl LintcheckConfig {
    pub fn new() -> Self {
        let clap_config = get_clap_config();

        // first, check if we got anything passed via the LINTCHECK_TOML env var,
        // if not, ask clap if we got any value for --crates-toml  <foo>
        // if not, use the default "lintcheck/lintcheck_crates.toml"
        let sources_toml = env::var("LINTCHECK_TOML").unwrap_or_else(|_| {
            clap_config
                .value_of("crates-toml")
                .clone()
                .unwrap_or("lintcheck/lintcheck_crates.toml")
                .to_string()
        });

        let markdown = clap_config.is_present("markdown");
        let sources_toml_path = PathBuf::from(sources_toml);

        // for the path where we save the lint results, get the filename without extension (so for
        // wasd.toml, use "wasd"...)
        let filename: PathBuf = sources_toml_path.file_stem().unwrap().into();
        let lintcheck_results_path = PathBuf::from(format!(
            "lintcheck-logs/{}_logs.{}",
            filename.display(),
            if markdown { "md" } else { "txt" }
        ));

        // look at the --threads arg, if 0 is passed, ask rayon rayon how many threads it would spawn and
        // use half of that for the physical core count
        // by default use a single thread
        let max_jobs = match clap_config.value_of("threads") {
            Some(threads) => {
                let threads: usize = threads
                    .parse()
                    .unwrap_or_else(|_| panic!("Failed to parse '{}' to a digit", threads));
                if threads == 0 {
                    // automatic choice
                    // Rayon seems to return thread count so half that for core count
                    (rayon::current_num_threads() / 2) as usize
                } else {
                    threads
                }
            },
            // no -j passed, use a single thread
            None => 1,
        };

        let lint_filter: Vec<String> = clap_config
            .values_of("filter")
            .map(|iter| {
                iter.map(|lint_name| {
                    let mut filter = lint_name.replace('_', "-");
                    if !filter.starts_with("clippy::") {
                        filter.insert_str(0, "clippy::");
                    }
                    filter
                })
                .collect()
            })
            .unwrap_or_default();

        LintcheckConfig {
            max_jobs,
            sources_toml_path,
            lintcheck_results_path,
            only: clap_config.value_of("only").map(String::from),
            fix: clap_config.is_present("fix"),
            lint_filter,
            markdown,
        }
    }
}
