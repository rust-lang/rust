// rustfmt-edition: 2018

fn foo() -> impl async Fn() {}

fn bar() -> impl for<'a> async Fn(&'a ()) {}
