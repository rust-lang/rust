struct S;

impl S {
    static fn f() {}
    //~^ ERROR expected one of `async`, `const`, `crate`, `default`, `extern`, `fn`, `pub`, `type`,
}

fn main() {}
