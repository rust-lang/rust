


// xfail-stage0
fn int_id(x: int) -> int { ret x; }

fn main() { while true { int_id(break); } }