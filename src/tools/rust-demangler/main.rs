//! Demangles rustc mangled names.
//!
//! This tool uses https://crates.io/crates/rustc-demangle to convert an input buffer of
//! newline-separated mangled names into their demangled translations.
//!
//! This tool can be leveraged by other applications that support third-party demanglers.
//! It takes a list of mangled names (one per line) on standard input, and prints a corresponding
//! list of demangled names. The tool is designed to support other programs that can leverage a
//! third-party demangler, such as `llvm-cov`, via the `-Xdemangler=<path-to-demangler>` option.
//!
//! To use `rust-demangler`, first build the tool with:
//!
//! ```shell
//! $ ./x.py build rust-demangler
//! ```
//!
//! Then, with `llvm-cov` for example, add the `-Xdemangler=...` option:
//!
//! ```shell
//! $ TARGET="${PWD}/build/x86_64-unknown-linux-gnu"
//! $ "${TARGET}"/llvm/bin/llvm-cov show --Xdemangler="${TARGET}"/stage0-tools-bin/rust-demangler \
//!   --instr-profile=main.profdata ./main --show-line-counts-or-regions
//! ```

use regex::Regex;
use rustc_demangle::demangle;
use std::io::{self, Read, Write};

const REPLACE_COLONS: &str = "::";

fn main() -> io::Result<()> {
    let mut strip_crate_disambiguators = Some(Regex::new(r"\[[a-f0-9]{16}\]::").unwrap());

    let mut args = std::env::args();
    let progname = args.next().unwrap();
    for arg in args {
        if arg == "--disambiguators" || arg == "-d" {
            strip_crate_disambiguators = None;
        } else {
            eprintln!();
            eprintln!("Usage: {} [-d|--disambiguators]", progname);
            eprintln!();
            eprintln!(
                "This tool converts a list of Rust mangled symbols (one per line) into a\n
                corresponding list of demangled symbols."
            );
            eprintln!();
            eprintln!(
                "With -d (--disambiguators), Rust symbols mangled with the v0 symbol mangler may\n\
                include crate disambiguators (a 16 character hex value in square brackets).\n\
                Crate disambiguators are removed by default."
            );
            eprintln!();
            std::process::exit(1)
        }
    }

    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let lines = buffer.lines();
    let mut demangled_lines = Vec::new();
    for mangled in lines {
        let mut demangled = demangle(mangled).to_string();
        if let Some(re) = &strip_crate_disambiguators {
            demangled = re.replace_all(&demangled, REPLACE_COLONS).to_string();
        }
        demangled_lines.push(demangled);
    }
    demangled_lines.push("".to_string());
    io::stdout().write_all(demangled_lines.join("\n").as_bytes())?;
    Ok(())
}
