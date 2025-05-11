#![feature(never_type)]
#![feature(type_alias_impl_trait)]
#![feature(non_exhaustive_omitted_patterns_lint)]
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]
// Test that the lint traversal handles opaques correctly
#![deny(non_exhaustive_omitted_patterns)]

fn main() {}

#[derive(Copy, Clone)]
enum Void {}

fn return_never_rpit(x: Void) -> impl Copy {
    if false {
        match return_never_rpit(x) {
            _ => {} //~ ERROR unreachable
        }
    }
    x
}
fn friend_of_return_never_rpit(x: Void) {
    match return_never_rpit(x) {}
    //~^ ERROR non-empty
}

type T = impl Copy;
#[define_opaque(T)]
fn return_never_tait(x: Void) -> T {
    if false {
        match return_never_tait(x) {
            _ => {} //~ ERROR unreachable
        }
    }
    x
}
fn friend_of_return_never_tait(x: Void) {
    match return_never_tait(x) {}
    //~^ ERROR non-empty
}

fn option_never(x: Void) -> Option<impl Copy> {
    if false {
        match option_never(x) {
            None => {}
            Some(_) => {} //~ ERROR unreachable
        }
        match option_never(x) {
            None => {}
            _ => {} //~ ERROR unreachable
        }
    }
    Some(x)
}

fn option_never2(x: Void) -> impl Copy {
    if false {
        match option_never2(x) {
            None => {}
            Some(_) => {} //~ ERROR unreachable
        }
        match option_never2(x) {
            None => {}
            _ => {} //~ ERROR unreachable
        }
        match option_never2(x) {
            None => {}
        }
    }
    Some(x)
}

fn inner_never(x: Void) {
    type T = impl Copy;
    let y: T = x;
    match y {
        _ => {} //~ ERROR unreachable
    }
}

// This one caused ICE https://github.com/rust-lang/rust/issues/117100.
fn inner_tuple() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    match foo {
        _ => {}
        Some((a, b)) => {} //~ ERROR unreachable
    }
}

type U = impl Copy;
#[define_opaque(U)]
fn unify_never(x: Void, u: U) -> U {
    if false {
        match u {
            _ => {} //~ ERROR unreachable
        }
    }
    x
}

type V = impl Copy;
#[define_opaque(V)]
fn infer_in_match(x: Option<V>) {
    match x {
        None => {}
        Some((a, b)) => {}
        Some((mut x, mut y)) => {
            //~^ ERROR unreachable
            x = 42;
            y = "foo";
        }
    }
}

type W = impl Copy;
#[derive(Copy, Clone)]
struct Rec<'a> {
    n: u32,
    w: Option<&'a W>,
}
#[define_opaque(W)]
fn recursive_opaque() -> W {
    if false {
        match recursive_opaque() {
            // Check for the ol' ICE when the type is recursively opaque.
            _ => {}
            Rec { n: 0, w: Some(Rec { n: 0, w: _ }) } => {} //~ ERROR unreachable
        }
    }
    let w: Option<&'static W> = None;
    Rec { n: 0, w }
}

type X = impl Copy;
struct SecretelyVoid(X);
#[define_opaque(X)]
fn nested_empty_opaque(x: Void) -> X {
    if false {
        let opaque_void = nested_empty_opaque(x);
        let secretely_void = SecretelyVoid(opaque_void);
        match secretely_void {
            _ => {} //~ ERROR unreachable
        }
    }
    x
}

type Y = (impl Copy, impl Copy);
struct SecretelyDoubleVoid(Y);
#[define_opaque(Y)]
fn super_nested_empty_opaque(x: Void) -> Y {
    if false {
        let opaque_void = super_nested_empty_opaque(x);
        let secretely_void = SecretelyDoubleVoid(opaque_void);
        match secretely_void {
            _ => {} //~ ERROR unreachable
        }
    }
    (x, x)
}
