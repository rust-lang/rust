#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn method() {}
}

reuse Trait::*; //~ ERROR glob delegation is only supported in impls

trait OtherTrait {
    reuse Trait::*; //~ ERROR glob delegation is only supported in impls
}

extern "C" {
    reuse Trait::*; //~ ERROR delegation is not supported in `extern` blocks
}

fn main() {
    reuse Trait::*; //~ ERROR glob delegation is only supported in impls
}
