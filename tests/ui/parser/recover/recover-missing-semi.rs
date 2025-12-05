fn main() {
    let _: usize = ()
    //~^ ERROR mismatched types
    //~| ERROR expected `;`
    let _ = 3;
}

fn foo() -> usize {
    let _: usize = ()
    //~^ ERROR mismatched types
    //~| ERROR expected `;`
    return 3;
}
