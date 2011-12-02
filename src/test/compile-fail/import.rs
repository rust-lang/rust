// error-pattern: unresolved import
import zed::bar;
import zed::baz;
mod zed {
    fn bar() { log "bar"; }
}
fn main(args: [str]) { bar(); }
