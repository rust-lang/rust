//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.allowed-positions

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
const CONST: () = ();

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
static STATIC: () = ();

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
type Type = ();

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
enum Enum {}

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
impl Enum {}

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
extern "C" {}

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
fn fun() {}

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
struct Struct {}

#[diagnostic::do_not_recommend]
//~^WARN cannot be used on
trait Trait {}

#[diagnostic::do_not_recommend]
impl Trait for i32 {}

fn main() {}
