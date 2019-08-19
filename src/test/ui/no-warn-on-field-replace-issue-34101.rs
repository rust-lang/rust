// Issue 34101: Circa 2016-06-05, `fn inline` below issued an
// erroneous warning from the elaborate_drops pass about moving out of
// a field in `Foo`, which has a destructor (and thus cannot have
// content moved out of it). The reason that the warning is erroneous
// in this case is that we are doing a *replace*, not a move, of the
// content in question, and it is okay to replace fields within `Foo`.
//
// Another more subtle problem was that the elaborate_drops was
// creating a separate drop flag for that internally replaced content,
// even though the compiler should enforce an invariant that any drop
// flag for such subcontent of `Foo` will always have the same value
// as the drop flag for `Foo` itself.








// build-pass (FIXME(62277): could be check-pass?)

struct Foo(String);

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn inline() {
    // (dummy variable so `f` gets assigned `var1` in MIR for both fn's)
    let _s = ();
    let mut f = Foo(String::from("foo"));
    f.0 = String::from("bar");
}

fn outline() {
    let _s = String::from("foo");
    let mut f = Foo(_s);
    f.0 = String::from("bar");
}


fn main() {
    inline();
    outline();
}
