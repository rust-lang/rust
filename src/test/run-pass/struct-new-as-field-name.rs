struct Foo {
    new: int,
}

pub fn main() {
    let foo = Foo{ new: 3 };
    assert_eq!(foo.new, 3);
}
