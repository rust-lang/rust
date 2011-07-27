// error-pattern: unresolved modulename
import baz::zed::bar;
mod baz { }
mod zed {
    fn bar() { log "bar3"; }
}
fn main(args: vec[str]) { bar(); }