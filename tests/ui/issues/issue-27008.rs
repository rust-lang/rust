struct S;

fn main() {
    let b = [0; S];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `S`
}
