// NOTE: This doesn't recover anymore.

fn main() {
    let x = (1, 2, 3, 4);
    match x {
        (1, .., 4) => {}
        (1, .=., 4) => { let _: usize = ""; }
        //~^ ERROR expected pattern, found `.`
        (.=., 4) => {}
        (1, 2, 3, 4) => {}
    }
}
