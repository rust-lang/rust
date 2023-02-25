// compile-flags: -Ztrait-solver=next
// From #80800

trait SuperTrait {
    type A;
    type B;
}

trait Trait: SuperTrait<A = <Self as SuperTrait>::B> {}

fn transmute<A, B>(x: A) -> B {
    foo::<A, B, dyn Trait<A = A, B = B>>(x)
    //~^ ERROR type annotations needed: cannot satisfy `dyn Trait<A = A, B = B>: Trait`
}

fn foo<A, B, T: ?Sized>(x: T::A) -> B
where
    T: Trait<B = B>,
{
    x
}

static X: u8 = 0;
fn main() {
    let x = transmute::<&u8, &[u8; 1_000_000]>(&X);
    println!("{:?}", x[100_000]);
}
