//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use termcolor::WriteColor;

macro_rules! static_regex {
    ($re:literal) => {{
        static RE: ::std::sync::OnceLock<::regex::Regex> = ::std::sync::OnceLock::new();
        RE.get_or_init(|| ::regex::Regex::new($re).unwrap())
    }};
}

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
pub mod fluent_period;
mod fluent_used;
pub mod gcc_submodule;
pub(crate) mod iter_header;
pub mod known_bug;
pub mod mir_opt_tests;
pub mod pal;
pub mod rustdoc_css_themes;
pub mod rustdoc_gui_tests;
pub mod rustdoc_js;
pub mod rustdoc_templates;
pub mod style;
pub mod target_policy;
pub mod target_specific_tests;
pub mod tests_placement;
pub mod tests_revision_unpaired_stdout_stderr;
pub mod triagebot;
pub mod ui_tests;
pub mod unit_tests;
pub mod unknown_revision;
pub mod unstable_book;
pub mod walk;
pub mod x_version;
