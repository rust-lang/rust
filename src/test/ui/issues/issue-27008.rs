struct S;

fn main() {
    let b = [0; S];
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found type `S`
    //~| expected usize, found struct `S`
}
