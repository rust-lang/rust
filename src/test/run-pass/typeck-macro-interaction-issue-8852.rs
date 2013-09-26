enum T {
    A(int),
    B(f64)
}

macro_rules! test(
    ($e:expr) => (
        fn foo(a:T, b:T) -> T {
            match (a, b) {
                (A(x), A(y)) => A($e),
                (B(x), B(y)) => B($e),
                _ => fail2!()
            }
        }
    )
)

test!(x + y)

pub fn main() {
    foo(A(1), A(2));
}
