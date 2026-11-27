//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// See #124385 for more details.

trait Foo<'a> {}
fn needs_foo<T>(_: T)
where
    for<'a> Wrap<T>: Foo<'a>,
{
}

struct Wrap<T>(T);
impl<'a, T> Foo<'a> for Wrap<T> where T: Fn(&'a i32) {}

fn main() {
    needs_foo(|x| {
        //[current]~^ ERROR implementation of `Foo` is not general enough
        //[next]~^^ ERROR type annotations needed
        x.to_string();
    });
}
