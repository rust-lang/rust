pub mod old {
    pub type A = u8;
    pub type B<'a, T> = &'a T;
    pub type C<'a, T> = &'a T;
    pub type D<'a, T> = &'a T;
    pub type E<'a, T> = &'a T;
    pub type F<'a, U=u8> = &'a U;
    pub type G<'a, T> = &'a T;
}

pub mod new {
    pub type A = u16;
    pub type B<'a, 'b : 'a, T> = (&'a T, &'b T);
    pub type C<T> = T;
    pub type D<'a, T, U=Box<T>> = (&'a T, U);
    pub type E<'a, T, U> = (&'a T, U);
    pub type F<'a> = &'a u8;
    pub type G<'a> = (&'a u8);
}
