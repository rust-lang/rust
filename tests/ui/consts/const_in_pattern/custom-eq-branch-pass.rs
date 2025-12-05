//@ run-pass

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

const EMPTY: &[CustomEq] = &[];

fn main() {
    // BAR_BAZ itself is fine but the enum has other variants
    // that are non-structural. Still, this should be accepted.
    match Foo::Qux(CustomEq) {
        BAR_BAZ => panic!(),
        _ => {}
    }

    // Similarly, an empty slice of a type that is non-structural
    // is accepted.
    match &[CustomEq] as &[CustomEq] {
        EMPTY => panic!(),
        _ => {},
    }
}
