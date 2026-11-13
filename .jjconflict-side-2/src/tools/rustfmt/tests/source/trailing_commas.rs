// rustfmt-match_block_trailing_comma: true
// rustfmt-trailing_comma: Always

fn main() {
    match foo {
        x => {}
        y => {
            foo();
        }
        _ => x
    }
}

fn f<S, T>(x: T, y: S) -> T where T: P, S: Q
{
    x
}

impl Trait for T where T: P
{
    fn f(x: T) -> T where T: Q + R
    {
        x
    }
}

struct Pair<S, T> where T: P, S: P + Q {
    a: T,
    b: S
}

struct TupPair<S, T> (S, T) where T: P, S: P + Q;

enum E<S, T> where S: P, T: P {
    A {a: T},
}

type Double<T> where T: P, T: Q = Pair<T, T>;

extern "C" {
    fn f<S, T>(x: T, y: S) -> T where T: P, S: Q;
}

trait Q<S, T> where T: P, S: R
{
    fn f<U, V>(self, x: T, y: S, z: U) -> Self where U: P, V: P;
}
