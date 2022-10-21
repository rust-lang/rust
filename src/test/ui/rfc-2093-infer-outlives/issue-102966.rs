// check-pass
//
// This is the example from implicit_infer.rs.
//
// Without cycle detection, we get:
//
// | Round | New predicates                                                                 |
// | ----- | -------------------------------------------------------------------------------|
// |   (0) | Var has explicit bound R : 'var                                                |
// |     1 | Node gets <T as Trait<'node>>::Assoc: 'node                                    |
// |     1 | Var  gets <RGen<R> as Trait<'var>>::Assoc: 'var                                |
// |     2 | Node gets <RGen<<T as Trait<'node>>::Assoc> as Trait<'node>>::Assoc 'node      |
// |     2 | Var  gets <RGen<<RGen<R> as Trait<'var>>::Assoc> as Trait<'var>>::Assoc: 'var  |
// |   3.. | Goes on forever.                                                               |
// | ----- | -------------------------------------------------------------------------------|
//
// With cycle detection:
//
// | Round | New predicates                                                                 |
// | ----- | -------------------------------------------------------------------------------|
// |   (0) | Var has explicit bound R : 'var                                                |
// |     1 | Node gets  <T as Trait<'node>>::Assoc: 'node                                   |
// |     1 | Var  gets  <RGen<R> as Trait<'var>>::Assoc: 'var                               |
// |     2 | Node detects cycle and does not insert another substituted version             |
// | ----- | -------------------------------------------------------------------------------|
//

trait Trait<'a> {
    type Assoc: 'a;
}
struct Node<'node, T: Trait<'node>> {
    var: Var<'node, T::Assoc>,
    _use_r: Option<T::Assoc>,
}
struct Var<'var, R: 'var> {
    node: Box<Node<'var, RGen<R>>>,
}

struct RGen<R>(std::marker::PhantomData<R>);
impl<'a, R: 'a> Trait<'a> for RGen<R> {
    type Assoc = R;
}

fn main() {}
