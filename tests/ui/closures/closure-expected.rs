fn main() {
    let x = Some(1);
    let y = x.or_else(4);
    //~^ ERROR expected `FnOnce()` signature, found `{integer}`
}
