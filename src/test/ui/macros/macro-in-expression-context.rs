// check-pass

macro_rules! foo {
    () => {
        assert_eq!("A", "A");
        assert_eq!("B", "B");
    }
}

fn main() {
    foo!()
}
