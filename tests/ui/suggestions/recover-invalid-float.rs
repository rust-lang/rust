// run-rustfix

fn main() {
    let _: f32 = .3;
    //~^ ERROR float literals must have an integer part
    let _: f32 = .42f32;
    //~^ ERROR float literals must have an integer part
    let _: f64 = .5f64;
    //~^ ERROR float literals must have an integer part
}
