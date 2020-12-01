use std::collections::{HashMap, VecDeque};

fn main() {
    let sample = [1; 5];
    let indirect_iter = sample.iter().collect::<Vec<_>>();
    indirect_iter.into_iter().map(|x| (x, x + 1)).collect::<HashMap<_, _>>();
    let indirect_len = sample.iter().collect::<VecDeque<_>>();
    indirect_len.len();
    let indirect_empty = sample.iter().collect::<VecDeque<_>>();
    indirect_empty.is_empty();
    let indirect_contains = sample.iter().collect::<VecDeque<_>>();
    indirect_contains.contains(&&5);
    let indirect_negative = sample.iter().collect::<Vec<_>>();
    indirect_negative.len();
    indirect_negative
        .into_iter()
        .map(|x| (*x, *x + 1))
        .collect::<HashMap<_, _>>();

    // #6202
    let a = "a".to_string();
    let sample = vec![a.clone(), "b".to_string(), "c".to_string()];
    let non_copy_contains = sample.into_iter().collect::<Vec<_>>();
    non_copy_contains.contains(&a);
}
