//! This crate allows tools to enable rust logging without having to magically
//! match rustc's tracing crate version.

use std::env::{self, VarError};
use std::fmt::{self, Display};
use std::io;
use tracing_subscriber::filter::{Directive, EnvFilter, LevelFilter};
use tracing_subscriber::layer::SubscriberExt;

pub fn init_rustc_env_logger() -> Result<(), Error> {
    init_env_logger("RUSTC_LOG")
}

/// In contrast to `init_rustc_env_logger` this allows you to choose an env var
/// other than `RUSTC_LOG`.
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

    let layer = tracing_tree::HierarchicalLayer::default()
        .with_writer(io::stderr)
        .with_indent_lines(true)
        .with_ansi(color_logs)
        .with_targets(true)
        .with_indent_amount(2);
    #[cfg(parallel_compiler)]
    let layer = layer.with_thread_ids(true).with_thread_names(true);

    let subscriber = tracing_subscriber::Registry::default().with(filter).with(layer);
    tracing::subscriber::set_global_default(subscriber).unwrap();

    Ok(())
}

pub fn stdout_isatty() -> bool {
    atty::is(atty::Stream::Stdout)
}

pub fn stderr_isatty() -> bool {
    atty::is(atty::Stream::Stderr)
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
                "invalid log color value '{}': expected one of always, never, or auto",
                value,
            ),
            Error::NonUnicodeColorValue => write!(
                formatter,
                "non-Unicode log color value: expected one of always, never, or auto",
            ),
        }
    }
}
