// This test creates 2 constants that have the same contents but a different AllocId.
// When hashed, those two contants would have the same stable hash. From the point of view
// of the query system, the two calls to `try_destructure_mir_constant` would have the same
// `DepNode` but with different inputs.
//
// This test verifies that the query system does not ICE in such cases.
//
//@ revisions: cpass1 cpass2

#[derive(PartialEq, Eq)]
struct Id<'a> {
    ns: &'a str,
}
fn visit_struct() {
    let id = Id { ns: "random1" };
    const FLAG: Id<'static> = Id { ns: "needs_to_be_the_same" };
    match id {
        FLAG => {}
        _ => {}
    }
}
fn visit_struct2() {
    let id = Id { ns: "random2" };
    const FLAG: Id<'static> = Id { ns: "needs_to_be_the_same" };
    match id {
        FLAG => {}
        _ => {}
    }
}

fn main() {
    visit_struct();
    visit_struct2();
}
