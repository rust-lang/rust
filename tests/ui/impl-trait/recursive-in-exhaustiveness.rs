//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Test several spicy non-trivial recursive opaque definitions inferred from HIR typeck
// don't cause stack overflows in exhaustiveness code, which currently reveals opaques
// manually in a way that is not overflow aware.
//
// These should eventually be outright rejected, but today (some) non-trivial recursive
// opaque definitions are accepted, and changing that requires an FCP, so for now just
// make sure we don't stack overflow :^)

// Opaque<T> = Opaque<Opaque<T>>
//
// We unfortunately accept this today, and due to how opaque type relating is implemented
// in the NLL type relation, this defines `Opaque<T> = T`.
fn build<T>(x: T) -> impl Sized {
    //[current]~^ ERROR cannot resolve opaque type
    let (x,) = (build(x),);
    //[next]~^ ERROR type annotations needed
    build(x)
}

// Opaque<T> = (Opaque<T>,)
//
// Not allowed today. Detected as recursive.
fn build2<T>(x: T) -> impl Sized {
    //[current]~^ ERROR cannot resolve opaque type
    let (x,) = (build2(x),);
    //[next]~^ ERROR type annotations needed
    (build2(x),)
}

// Opaque<T> = Opaque<(T,)>
//
// Not allowed today. Detected as not defining.
fn build3<T>(x: T) -> impl Sized {
    //[current]~^ ERROR cannot resolve opaque type
    let (x,) = (build3((x,)),);
    build3(x)
    //[next]~^ ERROR type annotations needed
}

fn main() {}
