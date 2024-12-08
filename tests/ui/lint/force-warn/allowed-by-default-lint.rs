// --force-warn $LINT causes $LINT (which is allow-by-default) to warn
//@ compile-flags: --force-warn elided_lifetimes_in_paths
//@ check-pass

struct Foo<'a> {
    x: &'a u32,
}

fn foo(x: &Foo) {}
//~^ WARN hidden lifetime parameters in types are deprecated

fn main() {}
