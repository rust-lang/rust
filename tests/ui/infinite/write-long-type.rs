//~ ERROR overflow evaluating the requirement `Cons<_, _>: Flat`

// rustc-env:RUST_MIN_STACK=1048576
// normalize-stderr-test: "long-type-\d+" -> "long-type-hash"

// Test that printing long types doesn't overflow the stack.

#![recursion_limit = "2048"]
#![allow(unused)]
use std::marker::PhantomData;

struct Apple;

struct Nil;
struct Cons<Car, Cdr>(PhantomData<(Car, Cdr)>);

// ========= Concat =========

trait Concat {
    type Output;
}

impl<L2> Concat for (Nil, L2) {
    type Output = L2;
}

impl<Car, Cdr, L2> Concat for (Cons<Car, Cdr>, L2)
where
    (Cdr, L2): Concat, // Recursive step
{
    type Output = Cons<Car, <(Cdr, L2) as Concat>::Output>;
}

// ========= Flat =========

trait Flat {
    type Output;
}

impl Flat for Nil {
    type Output = Nil;
}

// Head is not Cons
impl<Head, Tail> Flat for Cons<Head, Tail>
where
    Tail: Flat,
{
    type Output = Cons<Head, <Tail as Flat>::Output>;
}

// Head is Cons
impl<HeadCar, HeadCdr, Tail> Flat for Cons<Cons<HeadCar, HeadCdr>, Tail>
where
    Cons<HeadCar, HeadCdr>: Flat,
    Tail: Flat,
    (<Cons<HeadCar, HeadCdr> as Flat>::Output, <Tail as Flat>::Output): Concat,
{
    type Output =
        <(<Cons<HeadCar, HeadCdr> as Flat>::Output, <Tail as Flat>::Output) as Concat>::Output;
}

type Computation = <Cons<Apple, Nil> as Flat>::Output;

fn main() {
    println!("{}", std::any::type_name::<Computation>());
}
