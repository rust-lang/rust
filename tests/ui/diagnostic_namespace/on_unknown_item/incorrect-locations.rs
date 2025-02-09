//@check-pass

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
const CONST: () = ();

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
static STATIC: () = ();

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
type Type = ();

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
enum Enum {}

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
impl Enum {}

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
extern "C" {}

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
fn fun() {}

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
struct Struct {}

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
trait Trait {}

#[diagnostic::on_unknown_item(message = "foo")]
//~^WARN `#[diagnostic::on_unknown_item]` can only be applied to `use` statements
impl Trait for i32 {}

#[diagnostic::on_unknown_item(message = "foo")]
use std::str::FromStr;

fn main() {}
