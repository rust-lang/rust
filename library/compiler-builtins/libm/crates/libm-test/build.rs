use std::fmt::Write;
use std::path::PathBuf;
use std::{env, fs};

fn main() {
    let cfg = Config::from_env();

    emit_optimization_cfg(&cfg);
    emit_cfg_shorthands(&cfg);
    list_all_tests(&cfg);
}

#[allow(dead_code)]
struct Config {
    manifest_dir: PathBuf,
    out_dir: PathBuf,
    opt_level: u8,
    target_arch: String,
    target_env: String,
    target_family: Option<String>,
    target_os: String,
    target_string: String,
    target_vendor: String,
    target_features: Vec<String>,
}

impl Config {
    fn from_env() -> Self {
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|feats| feats.split(',').map(ToOwned::to_owned).collect())
            .unwrap_or_default();

        Self {
            manifest_dir: PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()),
            out_dir: PathBuf::from(env::var("OUT_DIR").unwrap()),
            opt_level: env::var("OPT_LEVEL").unwrap().parse().unwrap(),
            target_arch: env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
            target_env: env::var("CARGO_CFG_TARGET_ENV").unwrap(),
            target_family: env::var("CARGO_CFG_TARGET_FAMILY").ok(),
            target_os: env::var("CARGO_CFG_TARGET_OS").unwrap(),
            target_string: env::var("TARGET").unwrap(),
            target_vendor: env::var("CARGO_CFG_TARGET_VENDOR").unwrap(),
            target_features,
        }
    }
}

/// Some tests are extremely slow. Emit a config option based on optimization level.
fn emit_optimization_cfg(cfg: &Config) {
    println!("cargo::rustc-check-cfg=cfg(optimizations_enabled)");

    if cfg.opt_level >= 2 {
        println!("cargo::rustc-cfg=optimizations_enabled");
    }
}

/// Provide an alias for common longer config combinations.
fn emit_cfg_shorthands(cfg: &Config) {
    println!("cargo::rustc-check-cfg=cfg(x86_no_sse)");
    if cfg.target_arch == "x86" && !cfg.target_features.iter().any(|f| f == "sse") {
        // Shorthand to detect i586 targets
        println!("cargo::rustc-cfg=x86_no_sse");
    }
}

/// Create a list of all source files in an array. This can be used for making sure that
/// all functions are tested or otherwise covered in some way.
// FIXME: it would probably be better to use rustdoc JSON output to get public functions.
fn list_all_tests(cfg: &Config) {
    let math_src = cfg.manifest_dir.join("../../src/math");

    let mut files = fs::read_dir(math_src)
        .unwrap()
        .map(|f| f.unwrap().path())
        .filter(|entry| entry.is_file())
        .map(|f| f.file_stem().unwrap().to_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    files.sort();

    let mut s = "pub const ALL_FUNCTIONS: &[&str] = &[".to_owned();
    for f in files {
        if f == "mod" {
            // skip mod.rs
            continue;
        }
        write!(s, "\"{f}\",").unwrap();
    }
    write!(s, "];").unwrap();

    let outfile = cfg.out_dir.join("all_files.rs");
    fs::write(outfile, s).unwrap();
}
