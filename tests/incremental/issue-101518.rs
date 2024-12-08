//@ revisions: cpass

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

fn main() {}
