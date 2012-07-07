// xfail-test
// error-pattern: unresolved
import zed::bar;
import zed::baz;
mod zed {
    fn bar() { #debug("bar"); }
}
fn main(args: ~[str]) { bar(); }
