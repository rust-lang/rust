const LENGTH: f64 = 2;
//~^ ERROR: mismatched types
//~| NOTE: expected `f64`, found integer
//~| NOTE: expected because

struct Thing {
    f: [[f64; 2]; LENGTH],
    //~^ ERROR: mismatched types
    //~| NOTE: expected `usize`, found `f64`
    //~| NOTE: array length
}

fn main() {
    let _t = Thing { f: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] };
}
