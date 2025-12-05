#![crate_type = "lib"]

// An example from #108704 of the linked-list cursor-like pattern of #46859/#48001, where the
// polonius alpha analysis shows the same imprecision as NLLs, unlike the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #108704
//@ [polonius] known-bug: #108704
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

struct Root {
    children: Vec<Node>,
}

struct Node {
    name: String,
    children: Vec<Node>,
}

fn merge_tree_ok(root: &mut Root, path: Vec<String>) {
    let mut elements = &mut root.children;

    for p in path.iter() {
        for (idx, el) in elements.iter_mut().enumerate() {
            if el.name == *p {
                elements = &mut elements[idx].children;
                break;
            }
        }
    }
}

// NLLs and polonius alpha fail here
fn merge_tree_ko(root: &mut Root, path: Vec<String>) {
    let mut elements = &mut root.children;

    for p in path.iter() {
        for (idx, el) in elements.iter_mut().enumerate() {
            if el.name == *p {
                elements = &mut el.children;
                break;
            }
        }
    }
}
