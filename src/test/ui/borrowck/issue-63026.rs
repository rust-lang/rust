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
    result //~ ERROR cannot return value referencing local binding
}

fn main() {}
