use crate::common::Config;
use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;

use tracing::*;

#[cfg(test)]
mod tests;

/// Conversion table from triple OS name to Rust SYSNAME
const OS_TABLE: &[(&str, &str)] = &[
    ("android", "android"),
    ("androideabi", "android"),
    ("cuda", "cuda"),
    ("darwin", "macos"),
    ("dragonfly", "dragonfly"),
    ("emscripten", "emscripten"),
    ("freebsd", "freebsd"),
    ("fuchsia", "fuchsia"),
    ("haiku", "haiku"),
    ("hermit", "hermit"),
    ("illumos", "illumos"),
    ("ios", "ios"),
    ("l4re", "l4re"),
    ("linux", "linux"),
    ("mingw32", "windows"),
    ("none", "none"),
    ("netbsd", "netbsd"),
    ("openbsd", "openbsd"),
    ("redox", "redox"),
    ("sgx", "sgx"),
    ("solaris", "solaris"),
    ("win32", "windows"),
    ("windows", "windows"),
    ("vxworks", "vxworks"),
];

const ARCH_TABLE: &[(&str, &str)] = &[
    ("aarch64", "aarch64"),
    ("aarch64_be", "aarch64"),
    ("amd64", "x86_64"),
    ("arm", "arm"),
    ("arm64", "aarch64"),
    ("armv4t", "arm"),
    ("armv5te", "arm"),
    ("armv7", "arm"),
    ("armv7s", "arm"),
    ("asmjs", "asmjs"),
    ("avr", "avr"),
    ("bpfeb", "bpf"),
    ("bpfel", "bpf"),
    ("hexagon", "hexagon"),
    ("i386", "x86"),
    ("i586", "x86"),
    ("i686", "x86"),
    ("m68k", "m68k"),
    ("mips", "mips"),
    ("mips64", "mips64"),
    ("mips64el", "mips64"),
    ("mipsisa32r6", "mips"),
    ("mipsisa32r6el", "mips"),
    ("mipsisa64r6", "mips64"),
    ("mipsisa64r6el", "mips64"),
    ("mipsel", "mips"),
    ("mipsisa32r6", "mips"),
    ("mipsisa32r6el", "mips"),
    ("mipsisa64r6", "mips64"),
    ("mipsisa64r6el", "mips64"),
    ("msp430", "msp430"),
    ("nvptx64", "nvptx64"),
    ("powerpc", "powerpc"),
    ("powerpc64", "powerpc64"),
    ("powerpc64le", "powerpc64"),
    ("riscv64gc", "riscv64"),
    ("s390x", "s390x"),
    ("sparc", "sparc"),
    ("sparc64", "sparc64"),
    ("sparcv9", "sparc64"),
    ("thumbv6m", "thumb"),
    ("thumbv7em", "thumb"),
    ("thumbv7m", "thumb"),
    ("wasm32", "wasm32"),
    ("x86_64", "x86_64"),
    ("xcore", "xcore"),
];

pub const ASAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-fuchsia",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-fuchsia",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
];

pub const LSAN_SUPPORTED_TARGETS: &[&str] = &[
    // FIXME: currently broken, see #88132
    // "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
];

pub const MSAN_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-unknown-linux-gnu", "x86_64-unknown-freebsd", "x86_64-unknown-linux-gnu"];

pub const TSAN_SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
];

pub const HWASAN_SUPPORTED_TARGETS: &[&str] =
    &["aarch64-linux-android", "aarch64-unknown-linux-gnu"];

const BIG_ENDIAN: &[&str] = &[
    "aarch64_be",
    "armebv7r",
    "mips",
    "mips64",
    "mipsisa32r6",
    "mipsisa64r6",
    "powerpc",
    "powerpc64",
    "s390x",
    "sparc",
    "sparc64",
    "sparcv9",
];

static ASM_SUPPORTED_ARCHS: &[&str] = &[
    "x86", "x86_64", "arm", "aarch64", "riscv32", "riscv64", "nvptx64", "hexagon", "mips",
    "mips64", "spirv", "wasm32",
];

pub fn has_asm_support(triple: &str) -> bool {
    ASM_SUPPORTED_ARCHS.contains(&get_arch(triple))
}

pub fn matches_os(triple: &str, name: &str) -> bool {
    // For the wasm32 bare target we ignore anything also ignored on emscripten
    // and then we also recognize `wasm32-bare` as the os for the target
    if triple == "wasm32-unknown-unknown" {
        return name == "emscripten" || name == "wasm32-bare";
    }
    let triple: Vec<_> = triple.split('-').collect();
    for &(triple_os, os) in OS_TABLE {
        if triple.contains(&triple_os) {
            return os == name;
        }
    }
    panic!("Cannot determine OS from triple");
}

/// Determine the architecture from `triple`
pub fn get_arch(triple: &str) -> &'static str {
    let triple: Vec<_> = triple.split('-').collect();
    for &(triple_arch, arch) in ARCH_TABLE {
        if triple.contains(&triple_arch) {
            return arch;
        }
    }
    panic!("Cannot determine Architecture from triple");
}

/// Determine the endianness from `triple`
pub fn is_big_endian(triple: &str) -> bool {
    let triple_arch = triple.split('-').next().unwrap();
    BIG_ENDIAN.contains(&triple_arch)
}

pub fn matches_env(triple: &str, name: &str) -> bool {
    if let Some(env) = triple.split('-').nth(3) { env.starts_with(name) } else { false }
}

pub fn get_pointer_width(triple: &str) -> &'static str {
    if (triple.contains("64") && !triple.ends_with("gnux32") && !triple.ends_with("gnu_ilp32"))
        || triple.starts_with("s390x")
    {
        "64bit"
    } else if triple.starts_with("avr") {
        "16bit"
    } else {
        "32bit"
    }
}

pub fn make_new_path(path: &str) -> String {
    assert!(cfg!(windows));
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match env::var(lib_path_env_var()) {
        Ok(curr) => format!("{}{}{}", path, path_div(), curr),
        Err(..) => path.to_owned(),
    }
}

pub fn lib_path_env_var() -> &'static str {
    "PATH"
}
fn path_div() -> &'static str {
    ";"
}

pub fn logv(config: &Config, s: String) {
    debug!("{}", s);
    if config.verbose {
        println!("{}", s);
    }
}

pub trait PathBufExt {
    /// Append an extension to the path, even if it already has one.
    fn with_extra_extension<S: AsRef<OsStr>>(&self, extension: S) -> PathBuf;
}

impl PathBufExt for PathBuf {
    fn with_extra_extension<S: AsRef<OsStr>>(&self, extension: S) -> PathBuf {
        if extension.as_ref().is_empty() {
            self.clone()
        } else {
            let mut fname = self.file_name().unwrap().to_os_string();
            if !extension.as_ref().to_str().unwrap().starts_with('.') {
                fname.push(".");
            }
            fname.push(extension);
            self.with_file_name(fname)
        }
    }
}
