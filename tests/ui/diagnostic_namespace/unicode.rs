#![crate_type = "lib"]
#![deny(malformed_diagnostic_format_literals)]

#[diagnostic::on_unimplemented(message = " \x00 \u{b123} \\\u{b123} {:?}")]
//~^ERROR positional arguments are not permitted in diagnostic attributes [malformed_diagnostic_format_literals]
//~|ERROR format specifiers are not permitted in diagnostic attributes [malformed_diagnostic_format_literals]
#[diagnostic::on_unimplemented(note = "🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀{:?}")]
//~^ERROR positional arguments are not permitted in diagnostic attributes [malformed_diagnostic_format_literals]
//~|ERROR format specifiers are not permitted in diagnostic attributes [malformed_diagnostic_format_literals]
pub trait ILoveUnicode {}
