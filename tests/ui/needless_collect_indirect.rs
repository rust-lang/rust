use std::collections::{BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};

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

    // Fix #5991
    let vec_a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let vec_b = vec_a.iter().collect::<Vec<_>>();
    if vec_b.len() > 3 {}
    let other_vec = vec![1, 3, 12, 4, 16, 2];
    let we_got_the_same_numbers = other_vec.iter().filter(|item| vec_b.contains(item)).collect::<Vec<_>>();

    // Fix #6297
    let sample = [1; 5];
    let multiple_indirect = sample.iter().collect::<Vec<_>>();
    let sample2 = vec![2, 3];
    if multiple_indirect.is_empty() {
        // do something
    } else {
        let found = sample2
            .iter()
            .filter(|i| multiple_indirect.iter().any(|s| **s % **i == 0))
            .collect::<Vec<_>>();
    }
}

mod issue7110 {
    // #7110 - lint for type annotation cases
    use super::*;

    fn lint_vec(string: &str) -> usize {
        let buffer: Vec<&str> = string.split('/').collect();
        buffer.len()
    }
    fn lint_vec_deque() -> usize {
        let sample = [1; 5];
        let indirect_len: VecDeque<_> = sample.iter().collect();
        indirect_len.len()
    }
    fn lint_linked_list() -> usize {
        let sample = [1; 5];
        let indirect_len: LinkedList<_> = sample.iter().collect();
        indirect_len.len()
    }
    fn lint_binary_heap() -> usize {
        let sample = [1; 5];
        let indirect_len: BinaryHeap<_> = sample.iter().collect();
        indirect_len.len()
    }
    fn dont_lint(string: &str) -> usize {
        let buffer: Vec<&str> = string.split('/').collect();
        for buff in &buffer {
            println!("{}", buff);
        }
        buffer.len()
    }
}

mod issue7975 {
    use super::*;

    fn direct_mapping_with_used_mutable_reference() -> Vec<()> {
        let test_vec: Vec<()> = vec![];
        let mut vec_2: Vec<()> = vec![];
        let mut_ref = &mut vec_2;
        let collected_vec: Vec<_> = test_vec.into_iter().map(|_| mut_ref.push(())).collect();
        collected_vec.into_iter().map(|_| mut_ref.push(())).collect()
    }

    fn indirectly_mapping_with_used_mutable_reference() -> Vec<()> {
        let test_vec: Vec<()> = vec![];
        let mut vec_2: Vec<()> = vec![];
        let mut_ref = &mut vec_2;
        let collected_vec: Vec<_> = test_vec.into_iter().map(|_| mut_ref.push(())).collect();
        let iter = collected_vec.into_iter();
        iter.map(|_| mut_ref.push(())).collect()
    }

    fn indirect_collect_after_indirect_mapping_with_used_mutable_reference() -> Vec<()> {
        let test_vec: Vec<()> = vec![];
        let mut vec_2: Vec<()> = vec![];
        let mut_ref = &mut vec_2;
        let collected_vec: Vec<_> = test_vec.into_iter().map(|_| mut_ref.push(())).collect();
        let iter = collected_vec.into_iter();
        let mapped_iter = iter.map(|_| mut_ref.push(()));
        mapped_iter.collect()
    }
}

fn allow_test() {
    #[allow(clippy::needless_collect)]
    let v = [1].iter().collect::<Vec<_>>();
    v.into_iter().collect::<HashSet<_>>();
}
