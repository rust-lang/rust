pub trait SecondTestTrait{}
pub trait TestTrait<T> {
    fn test<R: SecondTestTrait + ?Sized>(&self, rng: &mut R) -> T;
}
