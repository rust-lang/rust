#[diagnostic::on_unknown_item]
//~^WARN malformed `#[diagnostic::on_unknown_item]`
use std::str::FromStr;

#[diagnostic::on_unknown_item(foo = "bar", message = "foo")]
//~^WARN unknown `foo` option for `#[diagnostic::on_unknown_item]` attribute
use std::str::Bytes;

#[diagnostic::on_unknown_item(label = "foo", label = "bar")]
//~^WARN `label` is ignored due to previous definition of `label`
use std::str::Chars;

#[diagnostic::on_unknown_item(message = "Foo", message = "Bar")]
//~^WARN `message` is ignored due to previous definition of `message`
use std::str::NotExisting;
//~^ERROR Foo

fn main() {}
