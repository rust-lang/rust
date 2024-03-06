// Annnotations may contain projection types with inference variables as input.
// Make sure we don't get ambiguities when normalizing them.

//@ check-fail

// Single impl.
fn test1<A, B, C, D>(a: A, b: B, c: C) {
    trait Tr { type Ty; }
    impl<T: 'static> Tr for (T,) { type Ty = T; }

    let _: <(_,) as Tr>::Ty = a; //~ ERROR type `A`
    Some::<<(_,) as Tr>::Ty>(b); //~ ERROR type `B`
    || -> <(_,) as Tr>::Ty { c }; //~ ERROR type `C`
    |d: <(_,) as Tr>::Ty| -> D { d }; //~ ERROR type `D`
}


// Two impls. The selected impl depends on the actual type.
fn test2<A, B, C>(a: A, b: B, c: C) {
    trait Tr { type Ty; }
    impl<T: 'static> Tr for (u8, T) { type Ty = T; }
    impl<T>          Tr for (i8, T) { type Ty = T; }
    type Alias<X, Y> = (<(X, Y) as Tr>::Ty, X);

    fn temp() -> String { todo!() }

    // `u8` impl, requires static.
    let _: Alias<_, _> = (a, 0u8); //~ ERROR type `A`
    Some::<Alias<_, _>>((b, 0u8)); //~ ERROR type `B`
    || -> Alias<_, _> { (c, 0u8) }; //~ ERROR type `C`

    let _: Alias<_, _> = (&temp(), 0u8); //~ ERROR temporary value
    Some::<Alias<_, _>>((&temp(), 0u8)); //~ ERROR temporary value

    // `i8` impl, no region constraints.
    let _: Alias<_, _> = (&temp(), 0i8);
    Some::<Alias<_, _>>((&temp(), 0i8));
}

fn main() {}
