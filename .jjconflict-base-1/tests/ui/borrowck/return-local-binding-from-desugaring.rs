// To avoid leaking the names of local bindings from expressions like for loops, #60984
// explicitly ignored them, but an assertion that `LocalKind::Var` *must* have a name would
// trigger an ICE. Before this change, this file's output would be:
// ```
// error[E0515]: cannot return value referencing local variable `__next`
//   --> return-local-binding-from-desugaring.rs:LL:CC
//    |
// LL |     for ref x in xs {
//    |         ----- `__next` is borrowed here
// ...
// LL |     result
//    |     ^^^^^^ returns a value referencing data owned by the current function
// ```
// FIXME: ideally `LocalKind` would carry more information to more accurately explain the problem.

use std::collections::HashMap;
use std::hash::Hash;

fn group_by<I, F, T>(xs: &mut I, f: F) -> HashMap<T, Vec<&I::Item>>
where
    I: Iterator,
    F: Fn(&I::Item) -> T,
    T: Eq + Hash,
{
    let mut result = HashMap::new();
    for ref x in xs {
        let key = f(x);
        result.entry(key).or_insert(Vec::new()).push(x);
    }
    result //~ ERROR cannot return value referencing temporary value
}

fn main() {}
