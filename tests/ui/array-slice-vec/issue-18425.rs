//@ run-pass
// Check that codegen doesn't ICE when codegenning an array repeat
// expression with a count of 1 and a non-Copy element type.


fn main() {
    let _ = [Box::new(1_usize); 1];
}
