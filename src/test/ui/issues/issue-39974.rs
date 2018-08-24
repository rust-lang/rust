const LENGTH: f64 = 2;

struct Thing {
    f: [[f64; 2]; LENGTH],
    //~^ ERROR mismatched types
    //~| expected usize, found f64
}

fn main() {
    let _t = Thing { f: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] };
}
