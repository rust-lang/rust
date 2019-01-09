trait Foo {
    pub const Foo: u32;
    //~^ ERROR expected one of `async`, `const`, `extern`, `fn`, `type`, `unsafe`, or `}`, found
}

fn main() {}
