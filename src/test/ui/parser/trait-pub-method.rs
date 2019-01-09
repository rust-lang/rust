trait Foo {
    pub fn foo();
    //~^ ERROR expected one of `async`, `const`, `extern`, `fn`, `type`, `unsafe`, or `}`, found
}

fn main() {}
