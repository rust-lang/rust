fn main() {
    let x = Some(1);
    let y = x.or_else(4);
    //~^ ERROR expected an `FnOnce()` closure, found `{integer}`
}
