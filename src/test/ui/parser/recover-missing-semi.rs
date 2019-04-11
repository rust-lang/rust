fn main() {
    let _: usize = ()
    //~^ ERROR mismatched types
    let _ = 3;
    //~^ ERROR expected one of
}

fn foo() -> usize {
    let _: usize = ()
    //~^ ERROR mismatched types
    return 3;
    //~^ ERROR expected one of
}
