// run-pass

const C: &'static isize = &1000;
static D: isize = *C;

pub fn main() {
    assert_eq!(D, 1000);
}
