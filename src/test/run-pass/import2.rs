
import zed::bar;

mod zed {
    fn bar() { #debug("bar"); }
}

fn main() { bar(); }
