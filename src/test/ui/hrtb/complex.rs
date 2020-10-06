// check-pass

trait A<'a> {}
trait B<'b> {}
fn foo<T>() where for<'a> T: A<'a> + 'a {}
trait C<'c>: for<'a> A<'a> + for<'b> B<'b> {
    type As;
}
struct D<T> where T: for<'c> C<'c, As=&'c ()> {
    t: std::marker::PhantomData<T>,
}
trait E<'e> {
    type As;
}
trait F<'f>: for<'a> A<'a> + for<'e> E<'e> {}
struct G<T> where T: for<'f> F<'f, As=&'f ()> {
    t: std::marker::PhantomData<T>,
}

fn main() {}
