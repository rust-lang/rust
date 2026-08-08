// Checks that we perform WF checks on closure args, regardless of wether they
// are used in the closure it self.
// related to issue #104478

struct MyTy<T: Trait>(T);
trait Trait {}
impl Trait for &'static str {}
fn wf<T>(_: T) {}

fn main() {
    let _: for<'x> fn(MyTy<&'x str>) = |_| {}; //~ ERROR: lifetime may not live long enough

    let _: for<'x> fn(MyTy<&'x str>) = |x| wf(x); //~ ERROR: lifetime may not live long enough
}
