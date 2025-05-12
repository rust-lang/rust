const LENGTH: f64 = 2;
//~^ ERROR mismatched types
//~| NOTE expected `f64`, found integer

struct Thing {
    f: [[f64; 2]; LENGTH],
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `f64`
}

fn main() {
    let _t = Thing { f: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] };
}
