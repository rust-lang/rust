// run-pass
const FOO: isize = 42;

enum Bar {
    Boo = *[&FOO; 4][3],
}

fn main() {
    assert_eq!(Bar::Boo as isize, 42);
}
