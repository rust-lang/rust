fn map(x: Option<Box<()>>) -> Option<Box<()>> {
    match x {
        None => None,
        Some(x) => Some(x),
    }
}

fn main() {
    map(None);
}

// EMIT_MIR rustc.map.SimplifyLocals.diff
