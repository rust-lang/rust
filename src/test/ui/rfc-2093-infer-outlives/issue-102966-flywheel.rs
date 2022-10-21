// check-pass
//
// This is even thornier than issue-102966. It illustrates the need to track the whole history of
// substituting a predicate to detect cycles, not just the type it originated in (for example).
// Here, the cycle is found in Kind because the inference went [Map, Kind, Node, Var], and would
// continue [..., Kind, Node, Var, Kind, Node, Var, ...].
//

trait Trait<'a> {
    type Input: 'a;
    type Assoc: 'a;
}

// 3. Node picks up Kind's
//    <T as Trait<'kind>>::Input: 'kind,
//    and substitutes it as
//    <T as Trait<'node>>::Input: 'node.
//    Provenance: [Map, Kind, Node]
//
struct Node<'node, T: Trait<'node>>(Kind<'node, T>, Option<T::Assoc>);

enum Kind<'kind, T: Trait<'kind>> {
    // 2. Kind picks up Map's
    //    I                         : 'map,
    //    and substitutes it as
    //    <T as Trait<'kind>>::Input: 'kind.
    //    Provenance: [Map, Kind]
    Map(Map<'kind, T::Input, T::Assoc>),
    // 5. Kind picks up Var's <RGen<R> as Trait<'var >>::Input: 'var.
    //    The provenance is [Map, Kind, Node, Var]. Kind already appears;
    //    we have therefore found a cycle. So this predicate is discarded,
    //    Node won't pick it up in the next round, and inference will reach
    //    its fixed point and terminate.
    Var(Var<'kind, T::Assoc>),
}

struct RGen<R>(std::marker::PhantomData<R>);
impl<'a, R: 'a> Trait<'a> for RGen<R> {
    type Input = ();
    type Assoc = R;
}

// 4. Var picks up Node's <T       as Trait<'node>>::Input: 'node, and substitutes it as
//                        <RGen<R> as Trait<'var >>::Input: 'var.
//    Provenance: [Map, Kind, Node, Var]
struct Var<'var, R: 'var>(Box<Node<'var, RGen<R>>>);
// 1. The predicate I: 'map originates here, inferred because Map contains &'map I.
//    Provenance: [Map]
struct Map<'map, I, R>(&'map I, fn(I) -> R);

fn main() {}
