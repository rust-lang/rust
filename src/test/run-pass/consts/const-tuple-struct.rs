// run-pass

struct Bar(isize, isize);

static X: Bar = Bar(1, 2);

pub fn main() {
    match X {
        Bar(x, y) => {
            assert_eq!(x, 1);
            assert_eq!(y, 2);
        }
    }
}
