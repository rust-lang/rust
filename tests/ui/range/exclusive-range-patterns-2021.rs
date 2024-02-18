//@ edition:2021

fn main() {
    let n = 2;
    match n {
        0...3 => {}
        //~^ ERROR `...` range patterns are deprecated
        4...10 => {}
        //~^ ERROR `...` range patterns are deprecated
        (11...100) => {}
        //~^ ERROR `...` range patterns are deprecated
        _ => {}
    }
}
