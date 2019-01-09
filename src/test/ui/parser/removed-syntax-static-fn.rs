// ignore-tidy-linelength

struct S;

impl S {
    static fn f() {}
}
//~^^ ERROR expected one of `async`, `const`, `crate`, `default`, `existential`, `extern`, `fn`, `pub`, `type`,

fn main() {}
