#![deny(clippy::drain_collect)]
#![allow(dead_code)]

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

fn binaryheap(b: &mut BinaryHeap<i32>) -> BinaryHeap<i32> {
    b.drain().collect()
}

fn binaryheap_dont_lint(b: &mut BinaryHeap<i32>) -> HashSet<i32> {
    b.drain().collect()
}

fn hashmap(b: &mut HashMap<i32, i32>) -> HashMap<i32, i32> {
    b.drain().collect()
}

fn hashmap_dont_lint(b: &mut HashMap<i32, i32>) -> Vec<(i32, i32)> {
    b.drain().collect()
}

fn hashset(b: &mut HashSet<i32>) -> HashSet<i32> {
    b.drain().collect()
}

fn hashset_dont_lint(b: &mut HashSet<i32>) -> Vec<i32> {
    b.drain().collect()
}

fn vecdeque(b: &mut VecDeque<i32>) -> VecDeque<i32> {
    b.drain(..).collect()
}

fn vecdeque_dont_lint(b: &mut VecDeque<i32>) -> HashSet<i32> {
    b.drain(..).collect()
}

fn vec(b: &mut Vec<i32>) -> Vec<i32> {
    b.drain(..).collect()
}

fn vec2(b: &mut Vec<i32>) -> Vec<i32> {
    b.drain(0..).collect()
}

fn vec3(b: &mut Vec<i32>) -> Vec<i32> {
    b.drain(..b.len()).collect()
}

fn vec4(b: &mut Vec<i32>) -> Vec<i32> {
    b.drain(0..b.len()).collect()
}

fn vec_no_reborrow() -> Vec<i32> {
    let mut b = vec![1, 2, 3];
    b.drain(..).collect()
}

fn vec_dont_lint(b: &mut Vec<i32>) -> HashSet<i32> {
    b.drain(..).collect()
}

fn string(b: &mut String) -> String {
    b.drain(..).collect()
}

fn string_dont_lint(b: &mut String) -> HashSet<char> {
    b.drain(..).collect()
}

fn not_whole_length(v: &mut Vec<i32>) -> Vec<i32> {
    v.drain(1..).collect()
}

fn main() {}
