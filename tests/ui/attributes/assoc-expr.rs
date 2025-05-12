//@ check-pass
// This test triggered an assertion failure in token collection due to
// mishandling of attributes on associative expressions.

#![feature(cfg_eval)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(internal_features)]

fn main() {}

#[cfg_eval]
struct Foo1(
    [ bool; {
        let _x = 30;
        #[cfg_attr(unix, rustc_dummy(aa))] 1
    } ]
);

#[cfg_eval]
struct Foo12(
    [ bool; {
        let _x = 30;
        #[cfg_attr(unix, rustc_dummy(bb))] 1 + 2
    } ]
);

#[cfg_eval]
struct Foox(
    [ bool; {
        let _x = 30;
        #[cfg_attr(unix, rustc_dummy(cc))] _x
    } ]
);

#[cfg_eval]
struct Foox2(
    [ bool; {
        let _x = 30;
        #[cfg_attr(unix, rustc_dummy(dd))] _x + 2
    } ]
);
