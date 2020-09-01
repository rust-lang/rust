struct Foo {
    x: isize,
}
extern "C" {
    fn foo(1: ());
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn bar((): isize);
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn baz(Foo { x }: isize);
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn qux((x, y): ());
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn this_is_actually_ok(a: usize);
    fn and_so_is_this(_: usize);
}
fn main() {}
