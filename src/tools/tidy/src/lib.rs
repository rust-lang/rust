//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use termcolor::WriteColor;

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The expression that failed
/// * The error itself
/// * (optionally) a path connected to the error (e.g. failure to open a file)
#[macro_export]
macro_rules! t {
    ($e:expr, $p:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed on {} with {}", stringify!($e), ($p).display(), e),
        }
    };

    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
}

macro_rules! tidy_error {
    ($bad:expr, $($fmt:tt)*) => ({
        $crate::tidy_error(&format_args!($($fmt)*).to_string()).expect("failed to output error");
        *$bad = true;
    });
}

macro_rules! tidy_error_ext {
    ($tidy_error:path, $bad:expr, $($fmt:tt)*) => ({
        $tidy_error(&format_args!($($fmt)*).to_string()).expect("failed to output error");
        *$bad = true;
    });
}

fn tidy_error(args: &str) -> std::io::Result<()> {
    use std::io::Write;
    use termcolor::{Color, ColorChoice, ColorSpec, StandardStream};

    let mut stderr = StandardStream::stdout(ColorChoice::Auto);
    stderr.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;

    write!(&mut stderr, "tidy error")?;
    stderr.set_color(&ColorSpec::new())?;

    writeln!(&mut stderr, ": {args}")?;
    Ok(())
}

pub mod alphabetical;
pub mod bins;
pub mod debug_artifacts;
pub mod deps;
pub mod edition;
pub mod error_codes;
pub mod ext_tool_checks;
pub mod extdeps;
pub mod features;
pub mod fluent_alphabetical;
pub mod mir_opt_tests;
pub mod pal;
pub mod rustdoc_css_themes;
pub mod rustdoc_gui_tests;
pub mod style;
pub mod target_specific_tests;
pub mod tests_placement;
pub mod ui_tests;
pub mod unit_tests;
pub mod unstable_book;
pub mod walk;
pub mod x_version;
