//@ edition: 2018
#[cfg(accessible())] //~ ERROR: expected cfg-accessible path
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn foo() {}
#[cfg(accessible(42))] //~ ERROR: unsupported literal
fn foo() -> bool { true }
#[cfg(accessible(::std::boxed::Box))]
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn foo() -> bool { true }
#[cfg(not(accessible(::std::boxed::Box)))]
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn foo() -> bool { false }
#[cfg(accessible(::std::nonexistent::item, ::nonexistent2::item))] //~ ERROR: expected cfg-accessible path
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn bar() -> bool { false }
#[cfg(not(accessible(::std::nonexistent::item)))]
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn bar() -> bool { true }
#[cfg(accessible(::nonexistent_crate::item))] //~ ERROR: `cfg(accessible(..))` mentioned external crate is not accessible
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn baz() -> bool { false }
#[cfg(not(accessible(::nonexistent_crate::item)))] //~ ERROR: `cfg(accessible(..))` mentioned external crate is not accessible
//~^ ERROR `cfg(accessible(..))` is experimental and subject to change
fn baz() -> bool { true }

fn main() {
    assert!(foo());
    assert!(bar());
    baz(); //~ ERROR cannot find function
    assert!(cfg!(accessible(::std::boxed::Box))); //~ ERROR `cfg(accessible(..))` is experimental and subject to change
    assert!(!cfg!(accessible(::nonexistent::item)));  //~ ERROR: `cfg(accessible(..))` mentioned external crate is not accessible
    //~^ ERROR `cfg(accessible(..))` is experimental and subject to change
}
