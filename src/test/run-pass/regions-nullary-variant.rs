// pretty-expanded FIXME #23616

enum roption<'a> {
    a, b(&'a usize)
}

fn mk(cond: bool, ptr: &usize) -> roption {
    if cond {roption::a} else {roption::b(ptr)}
}

pub fn main() {}
