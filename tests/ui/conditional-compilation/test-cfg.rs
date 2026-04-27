//@ compile-flags: --cfg foo --check-cfg=cfg(foo,bar)
#![allow(unexpected_cfgs)]

#[cfg(all(foo, bar))] // foo AND bar
//~^ NOTE the item is gated behind `bar`
fn foo() {} //~ NOTE found an item that was configured out

#[cfg(feature = "meow")]
//~^ NOTE the item is gated behind the `meow` feature
fn bar() {} //~ NOTE found an item that was configured out

#[cfg(false)]
//~^ NOTE the item is disabled
fn baz() {} //~ NOTE found an item that was configured out

#[cfg(any(bar))]
//~^ NOTE the item is gated here
fn quux() {} //~ NOTE found an item that was configured out

#[cfg(any())]
//~^ NOTE the item is disabled
fn qsdf() {} //~ NOTE found an item that was configured out

fn main() {
    foo(); //~ ERROR cannot find function `foo` in this scope
    //~^ NOTE not found in this scope
    bar(); //~ ERROR cannot find function `bar` in this scope
    //~^ NOTE not found in this scope
    baz(); //~ ERROR cannot find function `baz` in this scope
    //~^ NOTE not found in this scope
    quux(); //~ ERROR cannot find function `quux` in this scope
    //~^ NOTE not found in this scope
    qsdf(); //~ ERROR cannot find function `qsdf` in this scope
    //~^ NOTE not found in this scope
}
