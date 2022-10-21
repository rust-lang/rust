// check-pass
trait Trait<'a> {
    type Assoc: 'a;
}
struct Node<'node, T: Trait<'node>>(Var<'node, T::Assoc>, Option<T::Assoc>);
struct RGen<R>(std::marker::PhantomData<R>);
impl<'a, R: 'a> Trait<'a> for RGen<R> {
    type Assoc = R;
}
struct Var<'var, R: 'var>(Box<Node<'var, RGen<R>>>);

fn main() {}
