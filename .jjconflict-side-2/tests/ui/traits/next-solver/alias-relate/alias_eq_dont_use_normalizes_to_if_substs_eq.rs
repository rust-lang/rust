//@ compile-flags: -Znext-solver

//@ check-pass
// (should not pass, should be turned into a coherence-only test)

// check that when computing `alias-eq(<() as Foo<u16, T>>::Assoc, <() as Foo<?0, T>>::Assoc)`
// we do not infer `?0 = u8` via the `for<STOP> (): Foo<u8, STOP>` impl or `?0 = u16` by
// relating substs as either could be a valid solution.

trait Foo<T, STOP> {
    type Assoc;
}

impl<STOP> Foo<u8, STOP> for ()
where
    (): Foo<u16, STOP>,
{
    type Assoc = <() as Foo<u16, STOP>>::Assoc;
}

impl Foo<u16, i8> for () {
    type Assoc = u8;
}

impl Foo<u16, i16> for () {
    type Assoc = u16;
}

fn output<T, U>() -> <() as Foo<T, U>>::Assoc
where
    (): Foo<T, U>,
{
    todo!()
}

fn incomplete<T>()
where
    (): Foo<u16, T>,
{
    // `<() as Foo<u16, STOP>>::Assoc == <() as Foo<_, STOP>>::Assoc`
    let _: <() as Foo<u16, T>>::Assoc = output::<_, T>();

    // let _: <() as Foo<u16, T>>::Assoc = output::<u8, T>(); // OK
    // let _: <() as Foo<u16, T>>::Assoc = output::<u16, T>(); // OK
}

fn main() {}
