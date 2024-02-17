//@ check-pass

trait A<'a> {}
trait B<'b> {}
fn foo<T>() where for<'a> T: A<'a> + 'a {}
trait C<'c>: for<'a> A<'a> + for<'b> B<'b> {
    type As;
}
struct D<T> where T: for<'c> C<'c, As=&'c ()> {
    t: std::marker::PhantomData<T>,
}
trait E<'e, 'g> {
    type As;
}
trait F<'f>: for<'a> A<'a> + for<'e> E<'e, 'f> {}
struct G<T> where T: for<'f> F<'f, As=&'f ()> {
    t: std::marker::PhantomData<T>,
}
trait H<'a, 'b> {
    type As;
}
trait I<'a>: for<'b> H<'a, 'b> {}

struct J<T> where T: for<'i> I<'i, As=&'i ()> {
    t: std::marker::PhantomData<T>,
}

fn main() {}
