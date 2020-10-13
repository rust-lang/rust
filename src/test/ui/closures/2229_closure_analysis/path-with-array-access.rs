#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

struct Node {
    id: i32,
    neighbours: [Option<Box<Node>>; 2],
}

fn main() {
    let mut node0_0 = Box::new(Node { id: 0, neighbours: Default::default() });
    let mut node0 = Box::new(Node { id: 0, neighbours: [Some(node0_0), None] });

    let mut node1 = Box::new(Node { id: 1, neighbours: [Some(node0), None] });

    let mut node2 = Box::new(Node { id: 1, neighbours: [Some(node1), None] });

    let c = #[rustc_capture_analysis]
    || {
        println!(
            "{}",
            node2.neighbours[0].unwrap().neighbours[0].unwrap().neighbours[0].unwrap().id
        );
    };

    c();
}
