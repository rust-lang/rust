use std::env;
use std::path::PathBuf;

use getopts::Options;

pub(crate) struct Config {
    pub(crate) nodejs: PathBuf,
    pub(crate) npm: PathBuf,
    pub(crate) rust_src: PathBuf,
    pub(crate) out_dir: PathBuf,
    pub(crate) initial_cargo: PathBuf,
    pub(crate) jobs: String,
    pub(crate) test_args: Vec<PathBuf>,
    pub(crate) goml_files: Vec<PathBuf>,
    pub(crate) rustc: PathBuf,
    pub(crate) rustdoc: PathBuf,
    pub(crate) verbose: bool,
}

impl Config {
    pub(crate) fn from_args(args: Vec<String>) -> Self {
        let mut opts = Options::new();
        opts.optopt("", "nodejs", "absolute path of nodejs", "PATH")
            .optopt("", "npm", "absolute path of npm", "PATH")
            .reqopt("", "out-dir", "output path of doc compilation", "PATH")
            .reqopt("", "rust-src", "root source of the rust source", "PATH")
            .reqopt(
                "",
                "initial-cargo",
                "path to cargo to use for compiling tests/rustdoc-gui/src/*",
                "PATH",
            )
            .reqopt("", "jobs", "jobs arg of browser-ui-test", "JOBS")
            .optflag("", "verbose", "run tests verbosely, showing all output")
            .optmulti("", "test-arg", "args for browser-ui-test", "FLAGS")
            .optmulti("", "goml-file", "goml files for testing with browser-ui-test", "LIST");

        let (argv0, args_) = args.split_first().unwrap();
        if args.len() == 1 || args[1] == "-h" || args[1] == "--help" {
            let message = format!("Usage: {} [OPTIONS] [TESTNAME...]", argv0);
            println!("{}", opts.usage(&message));
            std::process::exit(1);
        }

        let matches = &match opts.parse(args_) {
            Ok(m) => m,
            Err(f) => panic!("{:?}", f),
        };

        let Some(nodejs) = matches.opt_str("nodejs").map(PathBuf::from) else {
            eprintln!("`nodejs` was not provided. If not available, please install it");
            std::process::exit(1);
        };
        let Some(npm) = matches.opt_str("npm").map(PathBuf::from) else {
            eprintln!("`npm` was not provided. If not available, please install it");
            std::process::exit(1);
        };

        Self {
            nodejs,
            npm,
            rust_src: matches.opt_str("rust-src").map(PathBuf::from).unwrap(),
            out_dir: matches.opt_str("out-dir").map(PathBuf::from).unwrap(),
            initial_cargo: matches.opt_str("initial-cargo").map(PathBuf::from).unwrap(),
            jobs: matches.opt_str("jobs").unwrap(),
            goml_files: matches.opt_strs("goml-file").iter().map(PathBuf::from).collect(),
            test_args: matches.opt_strs("test-arg").iter().map(PathBuf::from).collect(),
            rustc: env::var("RUSTC").map(PathBuf::from).unwrap(),
            rustdoc: env::var("RUSTDOC").map(PathBuf::from).unwrap(),
            verbose: matches.opt_present("verbose"),
        }
    }
}
