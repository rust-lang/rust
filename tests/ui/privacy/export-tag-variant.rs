mod foo {
    pub fn x() { }

    enum Y { Y1 }
}

fn main() { let z = foo::Y::Y1; } //~ ERROR: enum `Y` is private
