trait A<'a, T> {
    fn f(&mut self) -> &'a mut T;
    fn p() -> T;
}
