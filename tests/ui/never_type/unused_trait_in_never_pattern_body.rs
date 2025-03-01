fn a() {
    match 0 {
        ! => || { //~ ERROR `!` patterns are experimental
        //~^ ERROR a never pattern is always unreachable
        //~^^ ERROR mismatched types
            use std::ops::Add;
            0.add(1)
        },
    }
}

fn main() {}
