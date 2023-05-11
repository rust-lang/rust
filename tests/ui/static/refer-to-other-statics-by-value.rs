// run-pass

static A: usize = 42;
static B: usize = A;

fn main() {
    assert_eq!(B, 42);
}
