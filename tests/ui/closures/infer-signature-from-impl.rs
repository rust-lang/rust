// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
//[next] known-bug: trait-system-refactor-initiative#71
//[current] check-pass

trait Foo {}
fn needs_foo<T>(_: T)
where
    Wrap<T>: Foo,
{
}

struct Wrap<T>(T);
impl<T> Foo for Wrap<T> where T: Fn(i32) {}

fn main() {
    needs_foo(|x| {
        x.to_string();
    });
}
