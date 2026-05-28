//@ run-rustfix

fn main() {
    for let _x of [1, 2, 3] {}
    //~^ ERROR expected pattern, found `let`
    //~| ERROR missing `in` in `for` loop

    match 1 {
        let 1 => {}
        //~^ ERROR expected pattern, found `let`
        _ => {}
    }
}
