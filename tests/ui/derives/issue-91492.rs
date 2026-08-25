// Reproduce the issue with vec
pub struct NoDerives;
fn fun1(foo: &mut Vec<NoDerives>, bar: &[NoDerives]) {
    foo.extend_from_slice(bar); //~ ERROR
}

// Reproduce the issue with vec
// and demonstrate that other derives are ignored in the suggested output
#[derive(Default, PartialEq)]
pub struct SomeDerives;
fn fun2(foo: &mut Vec<SomeDerives>, bar: &[SomeDerives]) {
    foo.extend_from_slice(bar); //~ ERROR
}

// Try and fail to reproduce the issue without vec.
// No idea why it doesnt reproduce the issue but its still a useful test case.
struct Object<T, A>(T, A);
impl<T: Clone, A: Default> Object<T, A> {
    fn use_clone(&self) {}
}
fn fun3(foo: Object<NoDerives, SomeDerives>) {
    foo.use_clone(); //~ ERROR
}

fn main() {}
