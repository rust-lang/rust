// compile-pass

#[deny(warnings)]

enum Empty { }
trait Bar<T> {}
impl Bar<Empty> for () {}

fn boo() -> impl Bar<Empty> {}

fn main() {
    boo();
}
