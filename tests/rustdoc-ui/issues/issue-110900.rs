//@ check-pass

#![crate_type="lib"]
#![feature(associated_type_bounds)]

trait A<'a> {}
trait B<'b> {}

trait C<'c>: for<'a> A<'a> + for<'b> B<'b> {
    type As;
}

trait E<'e> {
    type As;
}
trait F<'f>: for<'a> A<'a> + for<'e> E<'e> {}
struct G<T>
where
    T: for<'l, 'i> H<'l, 'i, As: for<'a> A<'a> + 'i>
{
    t: std::marker::PhantomData<T>,
}

trait I<'a, 'b, 'c> {
    type As;
}

trait H<'d, 'e>: for<'f> I<'d, 'f, 'e> + 'd {}
