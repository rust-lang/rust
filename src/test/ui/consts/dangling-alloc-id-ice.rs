// https://github.com/rust-lang/rust/issues/55223

union Foo<'a> {
    y: &'a (),
    long_live_the_unit: &'static (),
}

const FOO: &() = { //~ ERROR any use of this value will cause an error
    let y = ();
    unsafe { Foo { y: &y }.long_live_the_unit }
};

fn main() {}
