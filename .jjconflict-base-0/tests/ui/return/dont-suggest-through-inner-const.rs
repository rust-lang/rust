const fn f() -> usize {
    //~^ ERROR mismatched types
    const FIELD: usize = loop {
        0
        //~^ ERROR mismatched types
    };
}

fn main() {}
