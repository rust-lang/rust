use std::path::PathBuf;

#[derive(Default)]
struct Config {
    llvm_filecheck: Option<PathBuf>,
    filters: Vec<String>,
    rustc_flags: Vec<String>,
}

impl Config {
    fn new() -> Result<Self, String> {
        // We skip the program's name.
        let mut args = std::env::args().skip(1);
        let mut config = Self::default();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--llvm-filecheck" => {
                    config.llvm_filecheck = args.next().map(PathBuf::from);
                }
                "--filter" => {
                    if let Some(arg) = args.next() {
                        config.filters.push(arg);
                    }
                }
                "--" => {
                    config.rustc_flags.extend(&mut args);
                    // Nothing else to be read but the `break` makes it more clear.
                    break;
                }
                arg => return Err(format!("Unknown argument {arg:?}")),
            }
        }
        if config.llvm_filecheck.is_none() {
            Err("Missing `--llvm-filecheck` option".to_owned())
        } else if config.rustc_flags.is_empty() {
            Err("Missing rustc flags (passed after `--`)".to_owned())
        } else {
            Ok(config)
        }
    }
}

fn main() {
    let Config { llvm_filecheck, filters, rustc_flags } = match Config::new() {
        Ok(c) => c,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let mut test_config = compiletest_rs::Config::default();

    test_config.mode = compiletest_rs::common::Mode::Assembly;
    test_config.src_base = PathBuf::from("tests/asm");
    test_config.llvm_filecheck = llvm_filecheck;
    test_config.filters = filters;
    test_config.strict_headers = true;
    test_config.build_base = PathBuf::from("build/tests/asm");
    test_config.target_rustcflags = Some(rustc_flags.join(" "));
    test_config.link_deps();
    test_config.clean_rmeta();

    compiletest_rs::run_tests(&test_config)
}
