// Make sure that we don't accidentally collect an RPITIT hidden type that does not
// hold for all instantiations of the trait signature.

trait MkStatic {
    fn mk_static(self) -> &'static str;
}

impl MkStatic for &'static str {
    fn mk_static(self) -> &'static str { self }
}

trait Foo {
    fn foo<'a: 'static, 'late>(&'late self) -> impl MkStatic;
}

impl Foo for str {
    fn foo<'a: 'static>(&'a self) -> impl MkStatic + 'static {
    //~^ ERROR method not compatible with trait
        self
    }
}

fn call_foo<T: Foo + ?Sized>(t: &T) -> &'static str {
    t.foo().mk_static()
}

fn main() {
    let s = call_foo(String::from("hello, world").as_str());
    println!("> {s}");
}
