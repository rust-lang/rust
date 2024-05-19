fn foo(_: impl FnOnce(&u32) -> &u32) {
}

fn baz(_: impl FnOnce(&u32, u32) -> &u32) {
}

fn bar() {
    let x = 22;
    foo(|a| &x)
//~^ ERROR does not live long enough
}

fn foobar() {
    let y = 22;
    baz(|first, second| &y)
//~^ ERROR does not live long enough
}

fn main() { }
