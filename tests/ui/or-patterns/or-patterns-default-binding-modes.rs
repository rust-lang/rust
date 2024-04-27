// Test that or-patterns are pass-through with respect to default binding modes.

//@ check-pass

#![allow(irrefutable_let_patterns)]
#![allow(dropping_copy_types)]
#![allow(dropping_references)]

fn main() {
    // A regression test for a mistake we made at one point:
    match &1 {
        e @ &(1..=2) | e @ &(3..=4) => {}
        _ => {}
    }

    match &0 {
        0 | &1 => {}
        _ => {}
    }

    type R<'a> = &'a Result<u8, u8>;

    let res: R<'_> = &Ok(0);

    match res {
        // Alternatives propagate expected type / binding mode independently.
        Ok(mut x) | &Err(mut x) => drop::<u8>(x),
    }
    match res {
        &(Ok(x) | Err(x)) => drop::<u8>(x),
    }
    match res {
        Ok(x) | Err(x) => drop::<&u8>(x),
    }
    if let Ok(mut x) | &Err(mut x) = res {
        drop::<u8>(x);
    }
    if let &(Ok(x) | Err(x)) = res {
        drop::<u8>(x);
    }
    let (Ok(mut x) | &Err(mut x)) = res;
    drop::<u8>(x);
    let &(Ok(x) | Err(x)) = res;
    drop::<u8>(x);
    let (Ok(x) | Err(x)) = res;
    drop::<&u8>(x);
    for Ok(mut x) | &Err(mut x) in std::iter::once(res) {
        drop::<u8>(x);
    }
    for &(Ok(x) | Err(x)) in std::iter::once(res) {
        drop::<u8>(x);
    }
    for Ok(x) | Err(x) in std::iter::once(res) {
        drop::<&u8>(x);
    }
    fn f1((Ok(mut x) | &Err(mut x)): R<'_>) {
        drop::<u8>(x);
    }
    fn f2(&(Ok(x) | Err(x)): R<'_>) {
        drop::<u8>(x);
    }
    fn f3((Ok(x) | Err(x)): R<'_>) {
        drop::<&u8>(x);
    }

    // Wrap inside another type (a product for a simplity with irrefutable contexts).
    #[derive(Copy, Clone)]
    struct Wrap<T>(T);
    let wres = Wrap(res);

    match wres {
        Wrap(Ok(mut x) | &Err(mut x)) => drop::<u8>(x),
    }
    match wres {
        Wrap(&(Ok(x) | Err(x))) => drop::<u8>(x),
    }
    match wres {
        Wrap(Ok(x) | Err(x)) => drop::<&u8>(x),
    }
    if let Wrap(Ok(mut x) | &Err(mut x)) = wres {
        drop::<u8>(x);
    }
    if let Wrap(&(Ok(x) | Err(x))) = wres {
        drop::<u8>(x);
    }
    if let Wrap(Ok(x) | Err(x)) = wres {
        drop::<&u8>(x);
    }
    let Wrap(Ok(mut x) | &Err(mut x)) = wres;
    drop::<u8>(x);
    let Wrap(&(Ok(x) | Err(x))) = wres;
    drop::<u8>(x);
    let Wrap(Ok(x) | Err(x)) = wres;
    drop::<&u8>(x);
    for Wrap(Ok(mut x) | &Err(mut x)) in std::iter::once(wres) {
        drop::<u8>(x);
    }
    for Wrap(&(Ok(x) | Err(x))) in std::iter::once(wres) {
        drop::<u8>(x);
    }
    for Wrap(Ok(x) | Err(x)) in std::iter::once(wres) {
        drop::<&u8>(x);
    }
    fn fw1(Wrap(Ok(mut x) | &Err(mut x)): Wrap<R<'_>>) {
        drop::<u8>(x);
    }
    fn fw2(Wrap(&(Ok(x) | Err(x))): Wrap<R<'_>>) {
        drop::<u8>(x);
    }
    fn fw3(Wrap(Ok(x) | Err(x)): Wrap<R<'_>>) {
        drop::<&u8>(x);
    }

    // Nest some more:

    enum Tri<P> {
        A(P),
        B(P),
        C(P),
    }

    let tri = &Tri::A(&Ok(0));
    let (Tri::A(Ok(mut x) | Err(mut x))
    | Tri::B(&Ok(mut x) | Err(mut x))
    | &Tri::C(Ok(mut x) | Err(mut x))) = tri;
    drop::<u8>(x);

    match tri {
        Tri::A(Ok(mut x) | Err(mut x))
        | Tri::B(&Ok(mut x) | Err(mut x))
        | &Tri::C(Ok(mut x) | Err(mut x)) => drop::<u8>(x),
    }
}
