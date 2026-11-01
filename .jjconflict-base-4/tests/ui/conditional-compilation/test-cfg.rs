//@ compile-flags: --cfg foo --check-cfg=cfg(foo,bar)

#[cfg(all(foo, bar))] // foo AND bar
//~^ NOTE the item is gated here
fn foo() {} //~ NOTE found an item that was configured out

fn main() {
    foo(); //~ ERROR cannot find function `foo` in this scope
    //~^ NOTE not found in this scope
}
