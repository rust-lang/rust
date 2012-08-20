enum roption {
    a, b(&uint)
}

fn mk(cond: bool, ptr: &r/uint) -> roption/&r {
    if cond {a} else {b(ptr)}
}

fn main() {}