use foo::baz;
use bar::baz; //~ ERROR E0252

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}

fn main() {
}
