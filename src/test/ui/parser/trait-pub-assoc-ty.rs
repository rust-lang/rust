trait Foo {
    pub type Foo;
    //~^ ERROR expected one of `async`, `const`, `extern`, `fn`, `type`, `unsafe`, or `}`, found
}

fn main() {}
