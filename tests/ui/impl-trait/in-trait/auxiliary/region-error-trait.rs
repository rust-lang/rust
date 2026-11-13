pub trait Request {
    type A<'a>
    where
        Self: 'a;
    fn f(_: Self::A<'_>) -> impl Sized;
}
