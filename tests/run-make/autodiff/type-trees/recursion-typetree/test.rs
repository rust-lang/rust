#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// Self-referential struct to test recursion detection
#[derive(Clone)]
struct Node {
    value: f64,
    next: Option<Box<Node>>,
}

// Mutually recursive structs to test cycle detection
#[derive(Clone)]
struct GraphNodeA {
    value: f64,
    connections: Vec<GraphNodeB>,
}

#[derive(Clone)]
struct GraphNodeB {
    weight: f64,
    target: Option<Box<GraphNodeA>>,
}

#[autodiff_reverse(d_test_node, Duplicated, Active)]
#[no_mangle]
fn test_node(node: &Node) -> f64 {
    node.value * 2.0
}

#[autodiff_reverse(d_test_graph, Duplicated, Active)]
#[no_mangle]
fn test_graph(a: &GraphNodeA) -> f64 {
    a.value * 3.0
}

// Simple depth test - deeply nested but not circular
#[derive(Clone)]
struct Level1 {
    val: f64,
    next: Option<Box<Level2>>,
}
#[derive(Clone)]
struct Level2 {
    val: f64,
    next: Option<Box<Level3>>,
}
#[derive(Clone)]
struct Level3 {
    val: f64,
    next: Option<Box<Level4>>,
}
#[derive(Clone)]
struct Level4 {
    val: f64,
    next: Option<Box<Level5>>,
}
#[derive(Clone)]
struct Level5 {
    val: f64,
    next: Option<Box<Level6>>,
}
#[derive(Clone)]
struct Level6 {
    val: f64,
    next: Option<Box<Level7>>,
}
#[derive(Clone)]
struct Level7 {
    val: f64,
    next: Option<Box<Level8>>,
}
#[derive(Clone)]
struct Level8 {
    val: f64,
}

#[autodiff_reverse(d_test_deep, Duplicated, Active)]
#[no_mangle]
fn test_deep(deep: &Level1) -> f64 {
    deep.val * 4.0
}

fn main() {
    let node = Node { value: 1.0, next: None };

    let graph = GraphNodeA { value: 2.0, connections: vec![] };

    let deep = Level1 { val: 5.0, next: None };

    let mut d_node = Node { value: 0.0, next: None };

    let mut d_graph = GraphNodeA { value: 0.0, connections: vec![] };

    let mut d_deep = Level1 { val: 0.0, next: None };

    let _result1 = d_test_node(&node, &mut d_node, 1.0);
    let _result2 = d_test_graph(&graph, &mut d_graph, 1.0);
    let _result3 = d_test_deep(&deep, &mut d_deep, 1.0);
}
