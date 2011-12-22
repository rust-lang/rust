// error-pattern: unresolved import
import zed::bar;
import zed::baz;
mod zed {
    fn bar() { #debug("bar"); }
}
fn main(args: [str]) { bar(); }
