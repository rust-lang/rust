//@ edition: 2018
#![feature(cfg_accessible)]

#[cfg(accessible())] //~ ERROR: expected cfg-accessible path
fn foo() {}
#[cfg(accessible(42))] //~ ERROR: unsupported literal
fn foo() -> bool { true }
#[cfg(accessible(::std::boxed::Box))]
fn foo() -> bool { true }
#[cfg(not(accessible(::std::boxed::Box)))]
fn foo() -> bool { false }
#[cfg(accessible(::std::nonexistent::item, ::nonexistent2::item))] //~ ERROR: expected cfg-accessible path
fn bar() -> bool { false }
#[cfg(not(accessible(::std::nonexistent::item)))]
fn bar() -> bool { true }
#[cfg(accessible(::nonexistent_crate::item))] //~ ERROR: `cfg(accessible(..))` mentioned external crate is not accessible
fn baz() -> bool { false }
#[cfg(not(accessible(::nonexistent_crate::item)))] //~ ERROR: `cfg(accessible(..))` mentioned external crate is not accessible
fn baz() -> bool { true }

fn main() {
    assert!(foo());
    assert!(bar());
    baz(); //~ ERROR cannot find function
    assert!(cfg!(accessible(::std::boxed::Box)));
    assert!(!cfg!(accessible(::nonexistent::item)));  //~ ERROR: `cfg(accessible(..))` mentioned external crate is not accessible
}
