fn main() {
    let x = Some(1);
    let y = x.or_else(4);
    //~^ ERROR expected a `FnOnce<()>` closure, found `{integer}`
}
