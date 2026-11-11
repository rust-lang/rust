#![feature(decl_macro)]

mod m {
    macro mac() {}
}

fn main() {
    m::mac!(); //~ ERROR macro `mac` is private
}
