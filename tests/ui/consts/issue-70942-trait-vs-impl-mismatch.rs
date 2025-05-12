trait Nat {
    const VALUE: usize;
}

struct Zero;

impl Nat for Zero {
    const VALUE: i32 = 0;
    //~^ ERROR implemented const `VALUE` has an incompatible type for trait
}

fn main() {
    let _: [i32; Zero::VALUE] = [];
}
