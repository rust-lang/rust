//@ aux-crate:upstream=match_non_exhaustive_lib.rs
//@ dont-require-annotations: NOTE
//@ edition: 2024

#![deny(unreachable_patterns)]

#[non_exhaustive]
//~^ NOTE `#[non_exhaustive]` does not affect exhaustiveness checking within the defining crate
//~| NOTE `#[non_exhaustive]` does not affect exhaustiveness checking within the defining crate
//~| NOTE `#[non_exhaustive]` does not affect exhaustiveness checking within the defining crate
//~| NOTE `#[non_exhaustive]` does not affect exhaustiveness checking within the defining crate
enum E1 { A, B }

fn foo(e: E1) {
    match e {
        E1::A | E1::B => {}
        _ => {}
        //~^ ERROR unreachable pattern
    }
}

fn foo_ref(e: &E1) {
    match e {
        E1::A | E1::B => {}
        _ => {}
        //~^ ERROR unreachable pattern
    }
}

fn foo_mut_ref(e: &mut E1) {
    match e {
        E1::A | E1::B => {}
        _ => {}
        //~^ ERROR unreachable pattern
    }
}

fn bar(e: Option<E1>) {
    match e {
        Some(E1::A | E1::B | _) => {}
        //~^ ERROR unreachable pattern
        None => {}
    }
}

fn baz(e: upstream::E2) {
    match e {
        upstream::E2::A | upstream::E2::B => {}
        _ => {}
    }
}

fn main() {}
