fn main() {
    let _f: f32 = 0xAAf32;
    //~^ ERROR mismatched types
    //~| HELP rewrite this

    let _f: f32 = 0xAB_f32;
    //~^ ERROR mismatched types
    //~| HELP rewrite this

    let _f: f64 = 0xFF_f64;
    //~^ ERROR mismatched types
    //~| HELP rewrite this
}
