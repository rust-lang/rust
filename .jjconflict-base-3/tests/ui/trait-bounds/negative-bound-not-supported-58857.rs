// https://github.com/rust-lang/rust/issues/58857
struct Conj<A> {a : A}
trait Valid {}

impl<A: !Valid> Conj<A>{}
//~^ ERROR negative bounds are not supported

fn main() {}
