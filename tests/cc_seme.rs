#![feature(plugin)]
#![plugin(clippy)]

#[allow(dead_code)]
enum Baz {
    Baz1,
    Baz2,
}

struct Test {
    t: Option<usize>,
    b: Baz,
}

fn main() {
    use Baz::*;
    let x = Test { t: Some(0), b: Baz1 };

    match x {
        Test { t: Some(_), b: Baz1 } => unreachable!(),
        Test { t: Some(42), b: Baz2 } => unreachable!(),
        Test { t: None, .. } => unreachable!(),
        Test { .. } => unreachable!(),
    }
}
