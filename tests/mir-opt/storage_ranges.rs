// skip-filecheck
// EMIT_MIR storage_ranges.main.nll.0.mir

fn main() {
    let a = 0;
    {
        let b = &Some(a);
    }
    let c = 1;
}
