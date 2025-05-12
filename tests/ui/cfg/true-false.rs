//@ run-pass

#![feature(link_cfg)]

#[cfg(true)]
fn foo() -> bool {
    cfg!(true)
}

#[cfg(false)]
fn foo() -> bool {
    cfg!(false)
}

#[cfg_attr(true, cfg(false))]
fn foo() {}

#[link(name = "foo", cfg(false))]
extern "C" {}

fn main() {
    assert!(foo());
    assert!(cfg!(true));
    assert!(!cfg!(false));
    assert!(cfg!(not(false)));
    assert!(cfg!(all(true)));
    assert!(cfg!(any(true)));
    assert!(!cfg!(not(true)));
}
