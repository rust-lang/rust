#![feature(diagnostic_on_unknown)]
#[diagnostic::on_unknown]
//~^WARN missing options for `on_unknown` attribute
use std::str::FromStr;

#[diagnostic::on_unknown(foo = "bar", message = "foo")]
//~^WARN malformed `diagnostic::on_unknown` attribute
use std::str::Bytes;

#[diagnostic::on_unknown(label = "foo", label = "bar")]
//~^WARN `label` is ignored due to previous definition of `label`
use std::str::Chars;

#[diagnostic::on_unknown(message = "Foo", message = "Bar")]
//~^WARN `message` is ignored due to previous definition of `message`
use std::str::NotExisting;
//~^ERROR Foo

fn main() {}
