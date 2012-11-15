// error-pattern:failed to resolve import
use zed::bar;
use zed::baz;
mod zed {
    #[legacy_exports];
    fn bar() { debug!("bar"); }
}
fn main(args: ~[str]) { bar(); }
