//@ check-fail
//
// regression test for #68295

struct Matrix<R, C, S>(R, C, S);

impl<R, C, S> Matrix<R, C, S> {
    fn into_owned(self) -> Matrix<R, C, Owned<R, C, ()>>
    where
        (): Allocator<R, C>,
    {
        unimplemented!()
    }
}

impl<D, S> Matrix<D, D, S> {
    fn hermitian_part(&self) -> Matrix<D, D, Owned<D, D, ()>>
    where
        (): Allocator<D, D>,
    {
        unimplemented!()
    }
}

trait Allocator<R, C> {
    type Buffer;
}

trait Trait<R, C, A> {
    type Power;
}

impl<R, C, A: Allocator<R, C>> Trait<R, C, A> for () {
    type Power = A::Buffer;
}

type Owned<R, C, G> = <G as Trait<R, C, ()>>::Power;

fn crash<R, C>(input: Matrix<R, C, ()>) -> Matrix<R, C, u32>
where
    (): Allocator<R, C>,
{
    input.into_owned()
    //~^ ERROR mismatched types [E0308]
}

fn main() {}
