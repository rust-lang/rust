// check-pass
// edition:2018
// issue: https://github.com/rust-lang/rust/issues/97534

macro_rules! m {
    () => {
        macro_rules! foo {
            () => {}
        }
        use foo as bar;
    }
}

m!{}

use bar as baz;

baz!{}

macro_rules! foo2 {
    () => {};
}

macro_rules! m2 {
    () => {
        use foo2 as bar2;
    };
}

m2! {}

use bar2 as baz2;

baz2! {}

macro_rules! n1 {
    () => {
        macro_rules! n2 {
            () => {
                macro_rules! n3 {
                    () => {
                        macro_rules! n4 {
                            () => {}
                        }
                        use n4 as c4;
                    }
                }
                use n3 as c3;
            }
        }
        use n2 as c2;
    }
}

use n1 as c1;
c1!{}
use c2 as a2;
a2!{}
use c3 as a3;
a3!{}
use c4 as a4;
a4!{}

// https://github.com/rust-lang/rust/pull/108729#issuecomment-1474750675
// reversed
use d5 as d6;
use d4 as d5;
use d3 as d4;
use d2 as d3;
use d1 as d2;
use foo2 as d1;
d6! {}

// mess
use f3 as f4;
f5! {}
use f1 as f2;
use f4 as f5;
use f2 as f3;
use foo2 as f1;

fn main() {
}
