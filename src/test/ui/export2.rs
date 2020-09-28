mod foo {
    pub fn x() { bar::x(); } //~ ERROR failed to resolve: use of undeclared crate or module `bar`
}

mod bar {
    fn x() { println!("x"); }

    pub fn y() { }
}

fn main() { foo::x(); }
