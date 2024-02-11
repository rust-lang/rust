use std::mem::ManuallyDrop;

trait Tr1 { type As1: Copy; }
trait Tr2 { type As2: Copy; }

struct S1;
#[derive(Copy, Clone)]
struct S2;
impl Tr1 for S1 { type As1 = S2; }

trait _Tr3 {
    type A: Iterator<Item: Copy>;
    //~^ ERROR associated type bounds are unstable

    type B: Iterator<Item: 'static>;
    //~^ ERROR associated type bounds are unstable
}

struct _St1<T: Tr1<As1: Tr2>> {
//~^ ERROR associated type bounds are unstable
    outest: T,
    outer: T::As1,
    inner: <T::As1 as Tr2>::As2,
}

enum _En1<T: Tr1<As1: Tr2>> {
//~^ ERROR associated type bounds are unstable
    Outest(T),
    Outer(T::As1),
    Inner(<T::As1 as Tr2>::As2),
}

union _Un1<T: Tr1<As1: Tr2>> {
//~^ ERROR associated type bounds are unstable
    outest: ManuallyDrop<T>,
    outer: ManuallyDrop<T::As1>,
    inner: ManuallyDrop<<T::As1 as Tr2>::As2>,
}

type _TaWhere1<T> where T: Iterator<Item: Copy> = T;
//~^ ERROR associated type bounds are unstable

fn _apit(_: impl Tr1<As1: Copy>) {}
//~^ ERROR associated type bounds are unstable

fn _rpit() -> impl Tr1<As1: Copy> { S1 }
//~^ ERROR associated type bounds are unstable

const _cdef: impl Tr1<As1: Copy> = S1;
//~^ ERROR associated type bounds are unstable
//~| ERROR `impl Trait` is not allowed in const types

static _sdef: impl Tr1<As1: Copy> = S1;
//~^ ERROR associated type bounds are unstable
//~| ERROR `impl Trait` is not allowed in static types

fn main() {
    let _: impl Tr1<As1: Copy> = S1;
    //~^ ERROR associated type bounds are unstable
    //~| ERROR `impl Trait` is not allowed in the type of variable bindings
}
