//@ compile-flags: -Zincremental-verify-ich=yes
// issue: rust-lang/rust#83085 incremental ICE: forcing query with already existing `DepNode`
// this used to fail to build straight away without needing any kind of
// stage1/2 builds but tidy demands it
//@ revisions:rpass1 rpass2

fn main() {
    const BOO: &[u8; 0] = &[];
    match &[] {
        BOO => (),
        b"" => (),
        _ => (),
    }
}

#[derive(PartialEq, Eq)]
struct Id<'a> {
    ns: &'a str,
}
fn visit_struct() {
    let id = Id { ns: "random1" };
    const FLAG: Id<'static> = Id {
        ns: "needs_to_be_the_same",
    };
    match id {
        FLAG => {}
        _ => {}
    }
}
fn visit_struct2() {
    let id = Id { ns: "random2" };
    const FLAG: Id<'static> = Id {
        ns: "needs_to_be_the_same",
    };
    match id {
        FLAG => {}
        _ => {}
    }
}
