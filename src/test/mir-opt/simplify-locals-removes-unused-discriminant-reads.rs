// ignore-tidy-linelength
// compile-flags: -Zunsound-mir-opts

fn map(x: Option<Box<()>>) -> Option<Box<()>> {
    match x {
        None => None,
        Some(x) => Some(x),
    }
}

fn main() {
    map(None);
}

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// This test was broken by changes to enum deaggregation, and will be fixed when
// `SimplifyArmIdentity` is fixed more generally
// FIXME(JakobDegen) EMIT_MIR simplify_locals_removes_unused_discriminant_reads.map.SimplifyLocals.diff
