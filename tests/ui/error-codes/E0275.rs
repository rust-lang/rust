//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

//@ compile-flags: -Zwrite-long-types-to-disk=yes
trait Foo {}

struct Bar<T>(T);

impl<T> Foo for T where Bar<T>: Foo {}
//[current]~^ ERROR E0275

fn is_foo<T: Foo>() {}
//[current]~^ ERROR E0275
fn main() {
    is_foo::<()>();
    //[next]~^ ERROR E0275
}
