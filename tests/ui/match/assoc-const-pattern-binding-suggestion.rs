struct Scrutinee;

struct Inherent;
impl Inherent {
    const K: i32 = 0;
}

const FREE: i32 = 0;

fn qualified(source: Scrutinee) {
    match source {
        Inherent::K => {} //~ ERROR mismatched types
        _ => {}
    }
}

fn lone_ident(source: Scrutinee) {
    match source {
        FREE => {} //~ ERROR mismatched types
        _ => {}
    }
}

fn main() {}
