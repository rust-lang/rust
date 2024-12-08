//@ check-pass
#![feature(do_not_recommend)]

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
const CONST: () = ();

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
static STATIC: () = ();

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
type Type = ();

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
enum Enum {}

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
extern "C" {}

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
fn fun() {}

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
struct Struct {}

#[diagnostic::do_not_recommend]
//~^WARN `#[diagnostic::do_not_recommend]` can only be placed
trait Trait {}

#[diagnostic::do_not_recommend]
impl Trait for i32 {}

fn main() {}
