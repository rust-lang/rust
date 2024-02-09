// run-pass

#![warn(indirect_structural_match)]

struct CustomEq;

impl Eq for CustomEq {}
impl PartialEq for CustomEq {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

#[derive(PartialEq, Eq)]
#[allow(unused)]
enum Foo {
    Bar,
    Baz,
    Qux(CustomEq),
}

const BAR_BAZ: Foo = if 42 == 42 {
    Foo::Bar
} else {
    Foo::Qux(CustomEq) // dead arm
};

fn main() {
    match Foo::Qux(CustomEq) {
        BAR_BAZ => panic!(),
        _ => {}
    }
}
