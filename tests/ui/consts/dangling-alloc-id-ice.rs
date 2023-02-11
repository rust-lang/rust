// https://github.com/rust-lang/rust/issues/55223

union Foo<'a> {
    y: &'a (),
    long_live_the_unit: &'static (),
}

const FOO: &() = {
//~^ ERROR encountered dangling pointer in final constant
    let y = ();
    unsafe { Foo { y: &y }.long_live_the_unit }
};

fn main() {}
