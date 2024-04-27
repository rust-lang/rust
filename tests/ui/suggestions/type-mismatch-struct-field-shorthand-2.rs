struct RGB { r: f64, g: f64, b: f64 }

fn main() {
    let (r, g, c): (f32, f32, f32) = (0., 0., 0.);
    let _ = RGB { r, g, c };
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
    //~| ERROR struct `RGB` has no field named `c`
}
