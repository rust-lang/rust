//@ run-rustfix

pub enum struct Range {
    //~^ ERROR `enum` and `struct` are mutually exclusive
    Valid {
        begin: u32,
        len: u32,
    },
    Out,
}

fn main() {
}
