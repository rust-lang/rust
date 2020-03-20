// run-pass
// https://github.com/rust-lang/rust/issues/55223

union Foo<'a> {
    //~^ WARN union is never used
    y: &'a (),
    long_live_the_unit: &'static (),
}

const FOO: &() = {
    //~^ WARN constant item is never used
    let y = ();
    unsafe { Foo { y: &y }.long_live_the_unit }
};

fn main() {}
