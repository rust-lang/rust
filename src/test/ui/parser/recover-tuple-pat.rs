fn main() {
    let x = (1, 2, 3, 4);
    match x {
        (1, .., 4) => {}
        (1, .=., 4) => { let _: usize = ""; }
        //~^ ERROR expected pattern, found `.`
        //~| ERROR mismatched types
        (.=., 4) => {}
        //~^ ERROR expected pattern, found `.`
        (1, 2, 3, 4) => {}
    }
}
