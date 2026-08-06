// Regression test for #154720
//@ edition: 2018
//@ needs-unwind
//@ skip-filecheck

// EMIT_MIR unwind_in_vec.build-{closure#0}.built.after.mir
#[inline(never)]
pub async fn build(s: u64) -> Vec<String> {
    vec!["0".to_string(), "1".to_string(), "2".to_string(), "3".to_string(), "4".to_string()]
}
