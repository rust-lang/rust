//@ check-pass
#![allow(dead_code, unused_imports)]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
extern crate std as other_std;

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
const CONST: () = ();

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
static STATIC: () = ();

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
type Type = ();

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
enum Enum {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
impl Enum {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
extern "C" {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
fn fun() {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
struct Struct {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
trait Trait {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN cannot be used on
impl Trait for i32 {}

#[diagnostic::on_unknown(message = "foo")]
use std::str::FromStr;

fn main() {
}
