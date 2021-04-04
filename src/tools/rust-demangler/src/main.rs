//! Demangles rustc mangled names.
//!
//! Note regarding crate disambiguators:
//!
//! Some demangled symbol paths can include "crate disambiguator" suffixes, represented as a large
//! hexadecimal value enclosed in square braces, and appended to the name of the crate. a suffix to the
//! original crate name. For example, the `core` crate, here, includes a disambiguator:
//!
//! ```rust
//!     <generics::Firework<f64> as core[a7a74cee373f048]::ops::drop::Drop>::drop
//! ```
//!
//! These disambiguators are known to vary depending on environmental circumstances. As a result,
//! tests that compare results including demangled names can fail across development environments,
//! particularly with cross-platform testing. Also, the resulting crate paths are not syntactically
//! valid, and don't match the original source symbol paths, which can impact development tools.
//!
//! For these reasons, by default, `rust-demangler` uses a heuristic to remove crate disambiguators
//! from their original demangled representation before printing them to standard output. If crate
//! disambiguators are required, add the `-d` (or `--disambiguators`) flag, and the disambiguators
//! will not be removed.
//!
//! Also note that the disambiguators are stripped by a Regex pattern that is tolerant to some
//! variation in the number of hexadecimal digits. The disambiguators come from a hash value, which
//! typically generates a 16-digit hex representation on a 64-bit architecture; however, leading
//! zeros are not included, which can shorten the hex digit length, and a different hash algorithm
//! that might also be dependent on the architecture, might shorten the length even further. A
//! minimum length of 5 digits is assumed, which should be more than sufficient to support hex
//! representations that generate only 8-digits of precision with an extremely rare (but not
//! impossible) result with up to 3 leading zeros.
//!
//! Using a minimum number of digits less than 5 risks the possibility of stripping demangled name
//! components with a similar pattern. For example, some closures instantiated multiple times
//! include their own disambiguators, demangled as non-hashed zero-based indexes in square brackets.
//! These disambiguators seem to have more analytical value (for instance, in coverage analysis), so
//! they are not removed.

use rust_demangler::*;
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    // FIXME(richkadel): In Issue #77615 discussed updating the `rustc-demangle` library, to provide
    // an option to generate demangled names without including crate disambiguators. If that
    // happens, update this tool to use that option (if the `-d` flag is not set) instead stripping
    // them via the Regex heuristic. The update the doc comments and help.

    // Strip hashed hexadecimal crate disambiguators. Leading zeros are not enforced, and can be
    // different across different platform/architecture types, so while 16 hex digits are common,
    // they can also be shorter.
    //
    // Also note that a demangled symbol path may include the `[<digits>]` pattern, with zero-based
    // indexes (such as for closures, and possibly for types defined in anonymous scopes). Preferably
    // these should not be stripped.
    //
    // The minimum length of 5 digits supports the possibility that some target architecture (maybe
    // a 32-bit or smaller architecture) could generate a hash value with a maximum of 8 digits,
    // and more than three leading zeros should be extremely unlikely. Conversely, it should be
    // sufficient to assume the zero-based indexes for closures and anonymous scopes will never
    // exceed the value 9999.
    let mut strip_crate_disambiguators = Some(create_disambiguator_re());

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
                "This tool converts a list of Rust mangled symbols (one per line) into a\n\
                corresponding list of demangled symbols."
            );
            eprintln!();
            eprintln!(
                "With -d (--disambiguators), Rust symbols mangled with the v0 symbol mangler may\n\
                include crate disambiguators (a hexadecimal hash value, typically up to 16 digits\n\
                long, enclosed in square brackets)."
            );
            eprintln!();
            eprintln!(
                "By default, crate disambiguators are removed, using a heuristics-based regular\n\
                expression. (See the `rust-demangler` doc comments for more information.)"
            );
            eprintln!();
            std::process::exit(1)
        }
    }

    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let mut demangled_lines = demangle_lines(buffer.lines(), strip_crate_disambiguators);
    demangled_lines.push("".to_string()); // ensure a trailing newline
    io::stdout().write_all(demangled_lines.join("\n").as_bytes())?;
    Ok(())
}
