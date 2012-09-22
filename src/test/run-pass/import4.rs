
use zed::bar;

mod zed {
    #[legacy_exports];
    fn bar() { debug!("bar"); }
}

fn main(args: ~[~str]) { let zed = 42; bar(); }
