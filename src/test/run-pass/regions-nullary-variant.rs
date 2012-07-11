enum roption {
    a, b(&uint)
}

fn mk(cond: bool, ptr: &uint) -> roption {
    if cond {a} else {b(ptr)}
}

fn main() {}