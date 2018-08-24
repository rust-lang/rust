mod foo {
    pub fn x() { }

    enum y { y1, }
}

fn main() { let z = foo::y::y1; } //~ ERROR: enum `y` is private
