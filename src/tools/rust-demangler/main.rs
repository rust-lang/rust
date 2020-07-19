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

use rustc_demangle::demangle;
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let lines = buffer.lines();
    let mut demangled = Vec::new();
    for mangled in lines {
        demangled.push(demangle(mangled).to_string());
    }
    demangled.push("".to_string());
    io::stdout().write_all(demangled.join("\n").as_bytes())?;
    Ok(())
}
