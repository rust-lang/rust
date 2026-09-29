//@ run-pass

struct Node {
    next: Option<Box<Node>>,
}

fn a() {
    let mut node = Node { next: None };

    let src = &mut node;
    {
        src
    };
    src.next = None;
}

fn b() {
    let src = &mut (22, 44);
    {
        src
    };
    src.0 = 66;
}

fn main() {
    a();
    b();
}
