// error-pattern: unresolved modulename
import baz::zed::bar;
mod baz { }
mod zed {
    fn bar() { #debug("bar3"); }
}
fn main(args: [str]) { bar(); }
