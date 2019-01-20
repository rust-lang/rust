fn main() {
    let _: usize = .3;
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
    let _: usize = .42f32;
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
    let _: usize = .5f64;
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}
