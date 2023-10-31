struct RGB {
    g: f64,
    b: f64,
}

fn main() {
    let (r, alone_in_path, b): (f32, f32, f32) = (e.clone(), e.clone());
    //~^ ERROR cannot find value `e` in this scope
    //~| ERROR cannot find value `e` in this scope
    //~| ERROR mismatched types
    let _ = RGB { r, g, b };
    //~^ ERROR cannot find value `g` in this scope
    //~| ERROR struct `RGB` has no field named `r`
    //~| ERROR mismatched types
}
