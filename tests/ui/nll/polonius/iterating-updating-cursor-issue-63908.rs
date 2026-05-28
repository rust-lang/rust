#![crate_type = "lib"]

// An example from #63908 of the linked-list cursor-like pattern of #46859/#48001, where the
// polonius alpha analysis shows the same imprecision as NLLs, unlike the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #63908
//@ [polonius] known-bug: #63908
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

struct Node<T> {
    value: T,
    next: Option<Box<Self>>,
}

type List<T> = Option<Box<Node<T>>>;

fn remove_last_node_recursive<T>(node_ref: &mut List<T>) {
    let next_ref = &mut node_ref.as_mut().unwrap().next;

    if next_ref.is_some() {
        remove_last_node_recursive(next_ref);
    } else {
        *node_ref = None;
    }
}

// NLLs and polonius alpha fail here
fn remove_last_node_iterative<T>(mut node_ref: &mut List<T>) {
    loop {
        let next_ref = &mut node_ref.as_mut().unwrap().next;

        if next_ref.is_some() {
            node_ref = next_ref;
        } else {
            break;
        }
    }

    *node_ref = None;
}
