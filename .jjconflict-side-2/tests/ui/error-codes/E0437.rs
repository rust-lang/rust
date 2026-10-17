trait Foo {}

impl Foo for i32 {
    type Bar = bool; //~ ERROR E0437
}

fn main () {
}
