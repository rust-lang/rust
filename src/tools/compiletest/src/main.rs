use std::env;

use compiletest::common::{Config, Mode};
use compiletest::util::logv;
use compiletest::{parse_config, run_tests};

fn main() {
    tracing_subscriber::fmt::init();

    let config = parse_config(env::args().collect());

    if config.valgrind_path.is_none() && config.force_valgrind {
        panic!("Can't find Valgrind to run Valgrind tests");
    }

    if !config.has_tidy && config.mode == Mode::Rustdoc {
        eprintln!("warning: `tidy` is not installed; diffs will not be generated");
    }

    log_config(&config);
    run_tests(config);
}

fn log_config(config: &Config) {
    let c = config;
    logv(c, "configuration:".to_string());
    logv(c, format!("compile_lib_path: {:?}", config.compile_lib_path));
    logv(c, format!("run_lib_path: {:?}", config.run_lib_path));
    logv(c, format!("rustc_path: {:?}", config.rustc_path.display()));
    logv(c, format!("rustdoc_path: {:?}", config.rustdoc_path));
    logv(c, format!("rust_demangler_path: {:?}", config.rust_demangler_path));
    logv(c, format!("src_base: {:?}", config.src_base.display()));
    logv(c, format!("build_base: {:?}", config.build_base.display()));
    logv(c, format!("stage_id: {}", config.stage_id));
    logv(c, format!("mode: {}", config.mode));
    logv(c, format!("run_ignored: {}", config.run_ignored));
    logv(c, format!("filters: {:?}", config.filters));
    logv(c, format!("skip: {:?}", config.skip));
    logv(c, format!("filter_exact: {}", config.filter_exact));
    logv(
        c,
        format!("force_pass_mode: {}", opt_str(&config.force_pass_mode.map(|m| format!("{}", m))),),
    );
    logv(c, format!("runtool: {}", opt_str(&config.runtool)));
    logv(c, format!("host-rustcflags: {:?}", config.host_rustcflags));
    logv(c, format!("target-rustcflags: {:?}", config.target_rustcflags));
    logv(c, format!("target: {}", config.target));
    logv(c, format!("host: {}", config.host));
    logv(c, format!("android-cross-path: {:?}", config.android_cross_path.display()));
    logv(c, format!("adb_path: {:?}", config.adb_path));
    logv(c, format!("adb_test_dir: {:?}", config.adb_test_dir));
    logv(c, format!("adb_device_status: {}", config.adb_device_status));
    logv(c, format!("ar: {}", config.ar));
    logv(c, format!("linker: {:?}", config.linker));
    logv(c, format!("verbose: {}", config.verbose));
    logv(c, format!("quiet: {}", config.quiet));
    logv(c, "\n".to_string());
}

fn opt_str(maybestr: &Option<String>) -> &str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => s,
    }
}
