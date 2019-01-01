// See issue #55454.

// compile-pass

#[derive(PartialEq)]
struct This<T>(T);

fn main() {
    This(Some(&1)) == This(Some(&1));
}
