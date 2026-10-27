//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// This shows that it is not sound to treat lifetimes within trait bounds of
// aliases as not live. Lifetimes within the trait bounds can be used to create
// shorter lifetimes than the data within (for example, here, when the inner
// data is `'static`).

static STATIC: &str = "static";

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

trait Yield<'a> {
    fn get(&self) -> &'a str;
}
impl<'a, T: Captures<'a>> Yield<'a> for T {
    fn get(&self) -> &'a str {
        STATIC
    }
}

fn foo<'a>(_: &'a str) -> impl Sized + Captures<'a> + 'static {}

fn requires_static<T: 'static>(val: T) {}

fn main() {
    let o;
    {
        let s = String::from("hi");
        let r: &str = &s; //~ ERROR `s` does not live long enough
        o = foo(r);
    }
    let out: &str = o.get();
    requires_static(out);
}
