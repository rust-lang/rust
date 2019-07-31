// build-pass (FIXME(62277): could be check-pass?)

#[deny(warnings)]

enum Empty { }
trait Bar<T> {}
impl Bar<Empty> for () {}

fn boo() -> impl Bar<Empty> {}

fn main() {
    boo();
}
