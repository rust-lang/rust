#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// Create mutually recursive types that would cause cycles
#[repr(C)]
struct NodeA {
    value: f32,
    b_ref: Option<Box<NodeB>>,
}

#[repr(C)]
struct NodeB {
    value: f64,
    a_ref: Option<Box<NodeA>>, // Mutual recursion: A -> B -> A -> B...
}

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
fn test_recursion_depth(node: &NodeA) -> f32 {
    node.value
}

fn main() {
    let node = NodeA { value: 1.0, b_ref: None };
    let mut d_node = NodeA { value: 0.0, b_ref: None };
    let result = d_test(&node, &mut d_node, 1.0);
    std::hint::black_box(result);
}
