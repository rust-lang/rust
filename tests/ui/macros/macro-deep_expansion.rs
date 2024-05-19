//@ run-pass

macro_rules! foo2 {
    () => {
        "foo"
    }
}

macro_rules! foo {
    () => {
        foo2!()
    }
}

fn main() {
    assert_eq!(concat!(foo!(), "bar"), "foobar")
}
