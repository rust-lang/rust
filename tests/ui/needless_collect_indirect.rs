#![allow(clippy::uninlined_format_args, clippy::useless_vec)]
#![allow(clippy::needless_if, clippy::uninlined_format_args)]
#![warn(clippy::needless_collect)]

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

mod issue_8553 {
    fn test_for() {
        let vec = vec![1, 2];
        let w: Vec<usize> = vec.iter().map(|i| i * i).collect();

        for i in 0..2 {
            // Do not lint, because this method call is in the loop
            w.contains(&i);
        }

        for i in 0..2 {
            let y: Vec<usize> = vec.iter().map(|k| k * k).collect();
            let z: Vec<usize> = vec.iter().map(|k| k * k).collect();
            // Do lint
            y.contains(&i);
            for j in 0..2 {
                // Do not lint, because this method call is in the loop
                z.contains(&j);
            }
        }

        // Do not lint, because this variable is used.
        w.contains(&0);
    }

    fn test_while() {
        let vec = vec![1, 2];
        let x: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let mut n = 0;
        while n > 1 {
            // Do not lint, because this method call is in the loop
            x.contains(&n);
            n += 1;
        }

        while n > 2 {
            let y: Vec<usize> = vec.iter().map(|k| k * k).collect();
            let z: Vec<usize> = vec.iter().map(|k| k * k).collect();
            // Do lint
            y.contains(&n);
            n += 1;
            while n > 4 {
                // Do not lint, because this method call is in the loop
                z.contains(&n);
                n += 1;
            }
        }
    }

    fn test_loop() {
        let vec = vec![1, 2];
        let x: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let mut n = 0;
        loop {
            if n < 1 {
                // Do not lint, because this method call is in the loop
                x.contains(&n);
                n += 1;
            } else {
                break;
            }
        }

        loop {
            if n < 2 {
                let y: Vec<usize> = vec.iter().map(|k| k * k).collect();
                let z: Vec<usize> = vec.iter().map(|k| k * k).collect();
                // Do lint
                y.contains(&n);
                n += 1;
                loop {
                    if n < 4 {
                        // Do not lint, because this method call is in the loop
                        z.contains(&n);
                        n += 1;
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn test_while_let() {
        let vec = vec![1, 2];
        let x: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let optional = Some(0);
        let mut n = 0;
        while let Some(value) = optional {
            if n < 1 {
                // Do not lint, because this method call is in the loop
                x.contains(&n);
                n += 1;
            } else {
                break;
            }
        }

        while let Some(value) = optional {
            let y: Vec<usize> = vec.iter().map(|k| k * k).collect();
            let z: Vec<usize> = vec.iter().map(|k| k * k).collect();
            if n < 2 {
                // Do lint
                y.contains(&n);
                n += 1;
            } else {
                break;
            }

            while let Some(value) = optional {
                if n < 4 {
                    // Do not lint, because this method call is in the loop
                    z.contains(&n);
                    n += 1;
                } else {
                    break;
                }
            }
        }
    }

    fn test_if_cond() {
        let vec = vec![1, 2];
        let v: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let w = v.iter().collect::<Vec<_>>();
        // Do lint
        for _ in 0..w.len() {
            todo!();
        }
    }

    fn test_if_cond_false_case() {
        let vec = vec![1, 2];
        let v: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let w = v.iter().collect::<Vec<_>>();
        // Do not lint, because w is used.
        for _ in 0..w.len() {
            todo!();
        }

        w.len();
    }

    fn test_while_cond() {
        let mut vec = vec![1, 2];
        let mut v: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let mut w = v.iter().collect::<Vec<_>>();
        // Do lint
        while 1 == w.len() {
            todo!();
        }
    }

    fn test_while_cond_false_case() {
        let mut vec = vec![1, 2];
        let mut v: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let mut w = v.iter().collect::<Vec<_>>();
        // Do not lint, because w is used.
        while 1 == w.len() {
            todo!();
        }

        w.len();
    }

    fn test_while_let_cond() {
        let mut vec = vec![1, 2];
        let mut v: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let mut w = v.iter().collect::<Vec<_>>();
        // Do lint
        while let Some(i) = Some(w.len()) {
            todo!();
        }
    }

    fn test_while_let_cond_false_case() {
        let mut vec = vec![1, 2];
        let mut v: Vec<usize> = vec.iter().map(|i| i * i).collect();
        let mut w = v.iter().collect::<Vec<_>>();
        // Do not lint, because w is used.
        while let Some(i) = Some(w.len()) {
            todo!();
        }
        w.len();
    }
}
