//! This crate allows tools to enable rust logging without having to magically
//! match rustc's tracing crate version.
//!
//! For example if someone is working on rustc_ast and wants to write some
//! minimal code against it to run in a debugger, with access to the `debug!`
//! logs emitted by rustc_ast, that can be done by writing:
//!
//! ```toml
//! [dependencies]
//! rustc_ast = { path = "../rust/compiler/rustc_ast" }
//! rustc_log = { path = "../rust/compiler/rustc_log" }
//! rustc_span = { path = "../rust/compiler/rustc_span" }
//! ```
//!
//! ```
//! fn main() {
//!     rustc_log::init_env_logger("LOG").unwrap();
//!
//!     let edition = rustc_span::edition::Edition::Edition2021;
//!     rustc_span::create_session_globals_then(edition, || {
//!         /* ... */
//!     });
//! }
//! ```
//!
//! Now `LOG=debug cargo run` will run your minimal main.rs and show
//! rustc's debug logging. In a workflow like this, one might also add
//! `std::env::set_var("LOG", "debug")` to the top of main so that `cargo
//! run` by itself is sufficient to get logs.
//!
//! The reason rustc_log is a tiny separate crate, as opposed to exposing the
//! same things in rustc_driver only, is to enable the above workflow. If you
//! had to depend on rustc_driver in order to turn on rustc's debug logs, that's
//! an enormously bigger dependency tree; every change you make to rustc_ast (or
//! whichever piece of the compiler you are interested in) would involve
//! rebuilding all the rest of rustc up to rustc_driver in order to run your
//! main.rs. Whereas by depending only on rustc_log and the few crates you are
//! debugging, you can make changes inside those crates and quickly run main.rs
//! to read the debug logs.

#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![feature(is_terminal)]

use std::env::{self, VarError};
use std::fmt::{self, Display};
use std::io::{self, IsTerminal};
use tracing_core::{Event, Subscriber};
use tracing_subscriber::filter::{Directive, EnvFilter, LevelFilter};
use tracing_subscriber::fmt::{
    format::{self, FormatEvent, FormatFields},
    FmtContext,
};
use tracing_subscriber::layer::SubscriberExt;

pub fn init_env_logger(env: &str) -> Result<(), Error> {
    let filter = match env::var(env) {
        Ok(env) => EnvFilter::new(env),
        _ => EnvFilter::default().add_directive(Directive::from(LevelFilter::WARN)),
    };

    let color_logs = match env::var(String::from(env) + "_COLOR") {
        Ok(value) => match value.as_ref() {
            "always" => true,
            "never" => false,
            "auto" => stderr_isatty(),
            _ => return Err(Error::InvalidColorValue(value)),
        },
        Err(VarError::NotPresent) => stderr_isatty(),
        Err(VarError::NotUnicode(_value)) => return Err(Error::NonUnicodeColorValue),
    };

    let verbose_entry_exit = match env::var_os(String::from(env) + "_ENTRY_EXIT") {
        None => false,
        Some(v) => &v != "0",
    };

    let layer = tracing_tree::HierarchicalLayer::default()
        .with_writer(io::stderr)
        .with_indent_lines(true)
        .with_ansi(color_logs)
        .with_targets(true)
        .with_verbose_exit(verbose_entry_exit)
        .with_verbose_entry(verbose_entry_exit)
        .with_indent_amount(2);
    #[cfg(parallel_compiler)]
    let layer = layer.with_thread_ids(true).with_thread_names(true);

    let subscriber = tracing_subscriber::Registry::default().with(filter).with(layer);
    match env::var(format!("{env}_BACKTRACE")) {
        Ok(str) => {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_writer(io::stderr)
                .without_time()
                .event_format(BacktraceFormatter { backtrace_target: str.to_string() });
            let subscriber = subscriber.with(fmt_layer);
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
        Err(_) => {
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
    };

    Ok(())
}

struct BacktraceFormatter {
    backtrace_target: String,
}

impl<S, N> FormatEvent<S, N> for BacktraceFormatter
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let target = event.metadata().target();
        if !target.contains(&self.backtrace_target) {
            return Ok(());
        }
        let backtrace = std::backtrace::Backtrace::capture();
        writeln!(writer, "stack backtrace: \n{:?}", backtrace)
    }
}

pub fn stdout_isatty() -> bool {
    io::stdout().is_terminal()
}

pub fn stderr_isatty() -> bool {
    io::stderr().is_terminal()
}

#[derive(Debug)]
pub enum Error {
    InvalidColorValue(String),
    NonUnicodeColorValue,
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidColorValue(value) => write!(
                formatter,
                "invalid log color value '{value}': expected one of always, never, or auto",
            ),
            Error::NonUnicodeColorValue => write!(
                formatter,
                "non-Unicode log color value: expected one of always, never, or auto",
            ),
        }
    }
}
