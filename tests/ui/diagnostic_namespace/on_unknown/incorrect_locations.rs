//@ check-pass
#![allow(dead_code, unused_imports)]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
extern crate std as other_std;

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
const CONST: () = ();

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
static STATIC: () = ();

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
type Type = ();

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
enum Enum {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
impl Enum {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
extern "C" {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
fn fun() {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
struct Struct {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
trait Trait {}

#[diagnostic::on_unknown(message = "foo")]
//~^WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
impl Trait for i32 {}

#[diagnostic::on_unknown(message = "foo")]
use std::str::FromStr;

fn main() {
}
