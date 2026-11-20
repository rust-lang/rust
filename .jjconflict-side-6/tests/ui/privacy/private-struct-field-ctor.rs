mod a {
    pub struct Foo {
        x: isize
    }
}

fn main() {
    let s = a::Foo { x: 1 };    //~ ERROR field `x` of struct `Foo` is private
}
