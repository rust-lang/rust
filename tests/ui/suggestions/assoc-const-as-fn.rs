unsafe fn pointer(v: usize, w: u32) {}

pub trait UniformScalar {}
impl UniformScalar for u32 {}

pub trait GlUniformScalar: UniformScalar {
    const FACTORY: unsafe fn(usize, Self) -> ();
}
impl GlUniformScalar for u32 {
    const FACTORY: unsafe fn(usize, Self) -> () = pointer;
}

pub fn foo<T: UniformScalar>(value: T) {
    <T as GlUniformScalar>::FACTORY(1, value);
    //~^ ERROR the trait bound `T: GlUniformScalar` is not satisfied
}

fn main() {}
