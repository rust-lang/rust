// skip-filecheck
//@ test-mir-pass: SimplifyLocals-before-const-prop

fn map(x: Option<Box<()>>) -> Option<Box<()>> {
    match x {
        None => None,
        Some(x) => Some(x),
    }
}

fn main() {
    map(None);
}

// EMIT_MIR simplify_locals_removes_unused_discriminant_reads.map.SimplifyLocals-before-const-prop.diff
