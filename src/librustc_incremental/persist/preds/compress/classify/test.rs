use super::*;

#[test]
fn detect_cycles() {
    let (graph, nodes) = graph! {
        A -> C0,
        A -> C1,
        B -> C1,
        C0 -> C1,
        C1 -> C0,
        C0 -> D,
        C1 -> E,
    };
    let inputs = ["A", "B"];
    let outputs = ["D", "E"];
    let mut reduce = GraphReduce::new(&graph, |n| inputs.contains(n), |n| outputs.contains(n));
    Classify::new(&mut reduce).walk();

    assert!(!reduce.in_cycle(nodes("A"), nodes("C0")));
    assert!(!reduce.in_cycle(nodes("B"), nodes("C0")));
    assert!(reduce.in_cycle(nodes("C0"), nodes("C1")));
    assert!(!reduce.in_cycle(nodes("D"), nodes("C0")));
    assert!(!reduce.in_cycle(nodes("E"), nodes("C0")));
    assert!(!reduce.in_cycle(nodes("E"), nodes("A")));
}

/// Regr test for a bug where we forgot to pop nodes off of the stack
/// as we were walking. In this case, because edges are pushed to the front
/// of the list, we would visit OUT, then A, then IN, and then close IN (but forget
/// to POP. Then visit B, C, and then A, which would mark everything from A to C as
/// cycle. But since we failed to pop IN, the stack was `OUT, A, IN, B, C` so that
/// marked C and IN as being in a cycle.
#[test]
fn edge_order1() {
    let (graph, nodes) = graph! {
        A -> C,
        C -> B,
        B -> A,
        IN -> B,
        IN -> A,
        A -> OUT,
    };
    let inputs = ["IN"];
    let outputs = ["OUT"];
    let mut reduce = GraphReduce::new(&graph, |n| inputs.contains(n), |n| outputs.contains(n));
    Classify::new(&mut reduce).walk();

    assert!(reduce.in_cycle(nodes("B"), nodes("C")));

    assert!(!reduce.in_cycle(nodes("IN"), nodes("A")));
    assert!(!reduce.in_cycle(nodes("IN"), nodes("B")));
    assert!(!reduce.in_cycle(nodes("IN"), nodes("C")));
    assert!(!reduce.in_cycle(nodes("IN"), nodes("OUT")));
}

/// Same as `edge_order1` but in reverse order so as to detect a failure
/// if we were to enqueue edges onto end of list instead.
#[test]
fn edge_order2() {
    let (graph, nodes) = graph! {
        A -> OUT,
        IN -> A,
        IN -> B,
        B -> A,
        C -> B,
        A -> C,
    };
    let inputs = ["IN"];
    let outputs = ["OUT"];
    let mut reduce = GraphReduce::new(&graph, |n| inputs.contains(n), |n| outputs.contains(n));
    Classify::new(&mut reduce).walk();

    assert!(reduce.in_cycle(nodes("B"), nodes("C")));

    assert!(!reduce.in_cycle(nodes("IN"), nodes("A")));
    assert!(!reduce.in_cycle(nodes("IN"), nodes("B")));
    assert!(!reduce.in_cycle(nodes("IN"), nodes("C")));
    assert!(!reduce.in_cycle(nodes("IN"), nodes("OUT")));
}
