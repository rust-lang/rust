#![feature(never_type)]
#![feature(exhaustive_patterns)]

mod foo {
    pub struct SecretlyEmpty {
        _priv: !,
    }

    pub struct NotSoSecretlyEmpty {
        pub _pub: !,
    }
}

struct NotSoSecretlyEmpty {
    _priv: !,
}

enum Foo {
    A(foo::SecretlyEmpty),
    B(foo::NotSoSecretlyEmpty),
    C(NotSoSecretlyEmpty),
    D(u32, u32),
}

fn main() {
    let x: Foo = Foo::D(123, 456);
    let Foo::D(_y, _z) = x; //~ ERROR refutable pattern in local binding: `Foo::A(_)` not covered
}
