// xfail-test
// error-pattern: unresolved
use zed::bar;
use zed::baz;
mod zed {
    fn bar() { debug!("bar"); }
}
fn main(args: ~[str]) { bar(); }
