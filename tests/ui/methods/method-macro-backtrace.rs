// forbid-output: in this expansion of

macro_rules! make_method {
    ($name:ident) => ( fn $name(&self) { } )
}

struct S;

impl S {
    // We had a bug where these wouldn't clean up macro backtrace frames.
    make_method!(foo1);
    make_method!(foo2);
    make_method!(foo3);
    make_method!(foo4);
    make_method!(foo5);
    make_method!(foo6);
    make_method!(foo7);
    make_method!(foo8);

    // Cause an error. It shouldn't have any macro backtrace frames.
    fn bar(&self) { }
    fn bar(&self) { } //~ ERROR duplicate definitions
}

fn main() { }
