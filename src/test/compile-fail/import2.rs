// error-pattern: unresolved
use baz::zed::bar;
mod baz {
    #[legacy_exports]; }
mod zed {
    #[legacy_exports];
    fn bar() { debug!("bar3"); }
}
fn main(args: ~[str]) { bar(); }
