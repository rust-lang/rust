//@ revisions: normal exhaustive_patterns
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![feature(never_type)]

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
    //~^ NOTE `Foo` defined here
    A(foo::SecretlyEmpty),
    //~^ NOTE not covered
    B(foo::NotSoSecretlyEmpty),
    C(NotSoSecretlyEmpty),
    D(u32, u32),
}

fn main() {
    let x: Foo = Foo::D(123, 456);
    let Foo::D(_y, _z) = x;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE `Foo::A(_)` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern"
    //~| NOTE for more information
    //~| NOTE pattern `Foo::A(_)` is currently uninhabited
    //~| NOTE the matched value is of type `Foo`
    //~| HELP you might want to use `let else`
}
