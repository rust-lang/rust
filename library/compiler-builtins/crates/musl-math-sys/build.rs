use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{env, fs, str};

/// Static library that will be built
const LIB_NAME: &str = "musl_math_prefixed";

/// Files that have more than one symbol. Map of file names to the symbols defined in that file.
const MULTIPLE_SYMBOLS: &[(&str, &[&str])] = &[
    (
        "__invtrigl",
        &["__invtrigl", "__invtrigl_R", "__pio2_hi", "__pio2_lo"],
    ),
    ("__polevll", &["__polevll", "__p1evll"]),
    ("erf", &["erf", "erfc"]),
    ("erff", &["erff", "erfcf"]),
    ("erfl", &["erfl", "erfcl"]),
    ("exp10", &["exp10", "pow10"]),
    ("exp10f", &["exp10f", "pow10f"]),
    ("exp10l", &["exp10l", "pow10l"]),
    ("exp2f_data", &["exp2f_data", "__exp2f_data"]),
    ("exp_data", &["exp_data", "__exp_data"]),
    ("j0", &["j0", "y0"]),
    ("j0f", &["j0f", "y0f"]),
    ("j1", &["j1", "y1"]),
    ("j1f", &["j1f", "y1f"]),
    ("jn", &["jn", "yn"]),
    ("jnf", &["jnf", "ynf"]),
    ("lgamma", &["lgamma", "__lgamma_r"]),
    ("remainder", &["remainder", "drem"]),
    ("remainderf", &["remainderf", "dremf"]),
    ("lgammaf", &["lgammaf", "lgammaf_r", "__lgammaf_r"]),
    ("lgammal", &["lgammal", "lgammal_r", "__lgammal_r"]),
    ("log2_data", &["log2_data", "__log2_data"]),
    ("log2f_data", &["log2f_data", "__log2f_data"]),
    ("log_data", &["log_data", "__log_data"]),
    ("logf_data", &["logf_data", "__logf_data"]),
    ("pow_data", &["pow_data", "__pow_log_data"]),
    ("powf_data", &["powf_data", "__powf_log2_data"]),
    ("signgam", &["signgam", "__signgam"]),
    ("sqrt_data", &["sqrt_data", "__rsqrt_tab"]),
];

fn main() {
    let cfg = Config::from_env();

    if cfg.target_env == "msvc"
        || cfg.target_family == "wasm"
        || cfg.target_features.iter().any(|f| f == "thumb-mode")
    {
        println!(
            "cargo::warning=Musl doesn't compile with the current \
            target {}; skipping build",
            &cfg.target_string
        );
        return;
    }

    build_musl_math(&cfg);
}

#[allow(dead_code)]
#[derive(Debug)]
struct Config {
    manifest_dir: PathBuf,
    out_dir: PathBuf,
    musl_dir: PathBuf,
    musl_arch: String,
    target_arch: String,
    target_env: String,
    target_family: String,
    target_os: String,
    target_string: String,
    target_vendor: String,
    target_features: Vec<String>,
}

impl Config {
    fn from_env() -> Self {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|feats| feats.split(',').map(ToOwned::to_owned).collect())
            .unwrap_or_default();
        let musl_dir = manifest_dir.join("musl");

        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let musl_arch = if target_arch == "x86" {
            "i386".to_owned()
        } else {
            target_arch.clone()
        };

        println!(
            "cargo::rerun-if-changed={}/c_patches",
            manifest_dir.display()
        );
        println!("cargo::rerun-if-changed={}", musl_dir.display());

        Self {
            manifest_dir,
            out_dir: PathBuf::from(env::var("OUT_DIR").unwrap()),
            musl_dir,
            musl_arch,
            target_arch,
            target_env: env::var("CARGO_CFG_TARGET_ENV").unwrap(),
            target_family: env::var("CARGO_CFG_TARGET_FAMILY").unwrap(),
            target_os: env::var("CARGO_CFG_TARGET_OS").unwrap(),
            target_string: env::var("TARGET").unwrap(),
            target_vendor: env::var("CARGO_CFG_TARGET_VENDOR").unwrap(),
            target_features,
        }
    }
}

/// Build musl math symbols to a static library
fn build_musl_math(cfg: &Config) {
    let musl_dir = &cfg.musl_dir;
    let math = musl_dir.join("src/math");
    let arch_dir = musl_dir.join("arch").join(&cfg.musl_arch);
    assert!(
        math.exists(),
        "musl source not found. You may need to run `./ci/update-musl.sh`."
    );

    let source_map = find_math_source(&math, cfg);
    let out_path = cfg.out_dir.join(format!("lib{LIB_NAME}.a"));

    // Run configuration steps. Usually done as part of the musl `Makefile`.
    let obj_include = cfg.out_dir.join("musl_obj/include");
    fs::create_dir_all(&obj_include).unwrap();
    fs::create_dir_all(obj_include.join("bits")).unwrap();
    let sed_stat = Command::new("sed")
        .arg("-f")
        .arg(musl_dir.join("tools/mkalltypes.sed"))
        .arg(arch_dir.join("bits/alltypes.h.in"))
        .arg(musl_dir.join("include/alltypes.h.in"))
        .stderr(Stdio::inherit())
        .output()
        .unwrap();
    assert!(
        sed_stat.status.success(),
        "sed command failed: {:?}",
        sed_stat.status
    );

    fs::write(obj_include.join("bits/alltypes.h"), sed_stat.stdout).unwrap();

    let mut cbuild = cc::Build::new();
    cbuild
        .extra_warnings(false)
        .warnings(false)
        .flag_if_supported("-Wno-bitwise-op-parentheses")
        .flag_if_supported("-Wno-literal-range")
        .flag_if_supported("-Wno-parentheses")
        .flag_if_supported("-Wno-shift-count-overflow")
        .flag_if_supported("-Wno-shift-op-parentheses")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-std=c99")
        .flag_if_supported("-ffreestanding")
        .flag_if_supported("-nostdinc")
        .define("_ALL_SOURCE", "1")
        .define(
            "ROOT_INCLUDE_FEATURES",
            Some(musl_dir.join("include/features.h").to_str().unwrap()),
        )
        // Our overrides are in this directory
        .include(cfg.manifest_dir.join("c_patches"))
        .include(musl_dir.join("arch").join(&cfg.musl_arch))
        .include(musl_dir.join("arch/generic"))
        .include(musl_dir.join("src/include"))
        .include(musl_dir.join("src/internal"))
        .include(obj_include)
        .include(musl_dir.join("include"))
        .file(cfg.manifest_dir.join("c_patches/alias.c"));

    for (sym_name, src_file) in source_map {
        // Build the source file
        cbuild.file(src_file);

        // Trickery! Redefine the symbol names to have the prefix `musl_`, which allows us to
        // differentiate these symbols from whatever we provide.
        if let Some((_names, syms)) = MULTIPLE_SYMBOLS
            .iter()
            .find(|(name, _syms)| *name == sym_name)
        {
            // Handle the occasional file that defines multiple symbols
            for sym in *syms {
                cbuild.define(sym, Some(format!("musl_{sym}").as_str()));
            }
        } else {
            // If the file doesn't define multiple symbols, the file name will be the symbol
            cbuild.define(&sym_name, Some(format!("musl_{sym_name}").as_str()));
        }
    }

    if cfg!(windows) {
        // On Windows we don't have a good way to check symbols, so skip that step.
        cbuild.compile(LIB_NAME);
        return;
    }

    let objfiles = cbuild.compile_intermediates();

    // We create the archive ourselves with relocations rather than letting `cc` do it so we can
    // encourage it to resolve symbols now. This should help avoid accidentally linking the wrong
    // thing.
    let stat = cbuild
        .get_compiler()
        .to_command()
        .arg("-r")
        .arg("-o")
        .arg(&out_path)
        .args(objfiles)
        .status()
        .unwrap();
    assert!(stat.success());

    println!("cargo::rustc-link-lib={LIB_NAME}");
    println!("cargo::rustc-link-search=native={}", cfg.out_dir.display());

    validate_archive_symbols(&out_path);
}

/// Build a map of `name -> path`. `name` is typically the symbol name, but this doesn't account
/// for files that provide multiple symbols.
fn find_math_source(math_root: &Path, cfg: &Config) -> BTreeMap<String, PathBuf> {
    let mut map = BTreeMap::new();
    let mut arch_dir = None;

    // Locate all files and directories
    for item in fs::read_dir(math_root).unwrap() {
        let path = item.unwrap().path();
        let meta = fs::metadata(&path).unwrap();

        if meta.is_dir() {
            // Make note of the arch-specific directory if it exists
            if path.file_name().unwrap() == cfg.target_arch.as_str() {
                arch_dir = Some(path);
            }
            continue;
        }

        // Skip non-source files
        if path.extension().is_some_and(|ext| ext == "h") {
            continue;
        }

        let sym_name = path.file_stem().unwrap();
        map.insert(sym_name.to_str().unwrap().to_owned(), path.to_owned());
    }

    // If arch-specific versions are available, build those instead.
    if let Some(arch_dir) = arch_dir {
        for item in fs::read_dir(arch_dir).unwrap() {
            let path = item.unwrap().path();
            let sym_name = path.file_stem().unwrap();

            if path.extension().unwrap() == "s" {
                // FIXME: we never build assembly versions since we have no good way to
                // rename the symbol (our options are probably preprocessor or objcopy).
                continue;
            }
            map.insert(sym_name.to_str().unwrap().to_owned(), path);
        }
    }

    map
}

/// Make sure we don't have something like a loose unprefixed `_cos` called somewhere, which could
/// wind up linking to system libraries rather than the built musl library.
fn validate_archive_symbols(out_path: &Path) {
    const ALLOWED_UNDEF_PFX: &[&str] = &[
        // PIC and arch-specific
        ".TOC",
        "_GLOBAL_OFFSET_TABLE_",
        "__x86.get_pc_thunk",
        // gcc/compiler-rt/compiler-builtins symbols
        "__add",
        "__aeabi_",
        "__div",
        "__eq",
        "__extend",
        "__fix",
        "__float",
        "__gcc_",
        "__ge",
        "__gt",
        "__le",
        "__lshr",
        "__lt",
        "__mul",
        "__ne",
        "__stack_chk_fail",
        "__stack_chk_guard",
        "__sub",
        "__trunc",
        "__undef",
        // string routines
        "__bzero",
        "bzero",
        // FPENV interfaces
        "feclearexcept",
        "fegetround",
        "feraiseexcept",
        "fesetround",
        "fetestexcept",
    ];

    // List global undefined symbols
    let out = Command::new("nm")
        .arg("-guj")
        .arg(out_path)
        .stderr(Stdio::inherit())
        .output()
        .unwrap();

    let undef = str::from_utf8(&out.stdout).unwrap();
    let mut undef = undef.lines().collect::<Vec<_>>();
    undef.retain(|sym| {
        // Account for file formats that add a leading `_`
        !ALLOWED_UNDEF_PFX
            .iter()
            .any(|pfx| sym.starts_with(pfx) || sym[1..].starts_with(pfx))
    });

    assert!(
        undef.is_empty(),
        "found disallowed undefined symbols: {undef:#?}"
    );

    // Find any symbols that are missing the `_musl_` prefix`
    let out = Command::new("nm")
        .arg("-gUj")
        .arg(out_path)
        .stderr(Stdio::inherit())
        .output()
        .unwrap();

    let defined = str::from_utf8(&out.stdout).unwrap();
    let mut defined = defined.lines().collect::<Vec<_>>();
    defined.retain(|sym| {
        !(sym.starts_with("_musl_")
            || sym.starts_with("musl_")
            || sym.starts_with("__x86.get_pc_thunk"))
    });

    assert!(defined.is_empty(), "found unprefixed symbols: {defined:#?}");
}
