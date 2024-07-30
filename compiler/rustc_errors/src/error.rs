use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use rustc_error_messages::fluent_bundle::resolver::errors::{ReferenceKind, ResolverError};
use rustc_error_messages::{FluentArgs, FluentError};

#[derive(Debug)]
pub enum TranslateError<'args> {
    One {
        id: &'args Cow<'args, str>,
        args: &'args FluentArgs<'args>,
        kind: TranslateErrorKind<'args>,
    },
    Two {
        primary: Box<TranslateError<'args>>,
        fallback: Box<TranslateError<'args>>,
    },
}

impl<'args> TranslateError<'args> {
    pub fn message(id: &'args Cow<'args, str>, args: &'args FluentArgs<'args>) -> Self {
        Self::One { id, args, kind: TranslateErrorKind::MessageMissing }
    }

    pub fn primary(id: &'args Cow<'args, str>, args: &'args FluentArgs<'args>) -> Self {
        Self::One { id, args, kind: TranslateErrorKind::PrimaryBundleMissing }
    }

    pub fn attribute(
        id: &'args Cow<'args, str>,
        args: &'args FluentArgs<'args>,
        attr: &'args str,
    ) -> Self {
        Self::One { id, args, kind: TranslateErrorKind::AttributeMissing { attr } }
    }

    pub fn value(id: &'args Cow<'args, str>, args: &'args FluentArgs<'args>) -> Self {
        Self::One { id, args, kind: TranslateErrorKind::ValueMissing }
    }

    pub fn fluent(
        id: &'args Cow<'args, str>,
        args: &'args FluentArgs<'args>,
        errs: Vec<FluentError>,
    ) -> Self {
        Self::One { id, args, kind: TranslateErrorKind::Fluent { errs } }
    }

    pub fn and(self, fallback: TranslateError<'args>) -> TranslateError<'args> {
        Self::Two { primary: Box::new(self), fallback: Box::new(fallback) }
    }
}

#[derive(Debug)]
pub enum TranslateErrorKind<'args> {
    MessageMissing,
    PrimaryBundleMissing,
    AttributeMissing { attr: &'args str },
    ValueMissing,
    Fluent { errs: Vec<FluentError> },
}

impl fmt::Display for TranslateError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TranslateErrorKind::*;

        match self {
            Self::One { id, args, kind } => {
                writeln!(f, "failed while formatting fluent string `{id}`: ")?;
                match kind {
                    MessageMissing => writeln!(f, "message was missing")?,
                    PrimaryBundleMissing => writeln!(f, "the primary bundle was missing")?,
                    AttributeMissing { attr } => {
                        writeln!(f, "the attribute `{attr}` was missing")?;
                        writeln!(f, "help: add `.{attr} = <message>`")?;
                    }
                    ValueMissing => writeln!(f, "the value was missing")?,
                    Fluent { errs } => {
                        for err in errs {
                            match err {
                                FluentError::ResolverError(ResolverError::Reference(
                                    ReferenceKind::Message { id, .. }
                                    | ReferenceKind::Variable { id, .. },
                                )) => {
                                    if args.iter().any(|(arg_id, _)| arg_id == id) {
                                        writeln!(
                                            f,
                                            "argument `{id}` exists but was not referenced correctly"
                                        )?;
                                        writeln!(f, "help: try using `{{${id}}}` instead")?;
                                    } else {
                                        writeln!(
                                            f,
                                            "the fluent string has an argument `{id}` that was not found."
                                        )?;
                                        let vars: Vec<&str> =
                                            args.iter().map(|(a, _v)| a).collect();
                                        match &*vars {
                                            [] => writeln!(f, "help: no arguments are available")?,
                                            [one] => writeln!(
                                                f,
                                                "help: the argument `{one}` is available"
                                            )?,
                                            [first, middle @ .., last] => {
                                                write!(f, "help: the arguments `{first}`")?;
                                                for a in middle {
                                                    write!(f, ", `{a}`")?;
                                                }
                                                writeln!(f, " and `{last}` are available")?;
                                            }
                                        }
                                    }
                                }
                                _ => writeln!(f, "{err}")?,
                            }
                        }
                    }
                }
            }
            // If someone cares about primary bundles, they'll probably notice it's missing
            // regardless or will be using `debug_assertions`
            // so we skip the arm below this one to avoid confusing the regular user.
            Self::Two { primary: box Self::One { kind: PrimaryBundleMissing, .. }, fallback } => {
                fmt::Display::fmt(fallback, f)?;
            }
            Self::Two { primary, fallback } => {
                writeln!(
                    f,
                    "first, fluent formatting using the primary bundle failed:\n {primary}\n \
                    while attempting to recover by using the fallback bundle instead, another error occurred:\n{fallback}"
                )?;
            }
        }
        Ok(())
    }
}

impl Error for TranslateError<'_> {}
