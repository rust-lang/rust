// unit-test: SimplifyLocals

fn map(x: Option<Box<()>>) -> Option<Box<()>> {
    match x {
        None => None,
        Some(x) => Some(x),
    }
}

fn main() {
    map(None);
}

// EMIT_MIR simplify_locals_removes_unused_discriminant_reads.map.SimplifyLocals.diff
