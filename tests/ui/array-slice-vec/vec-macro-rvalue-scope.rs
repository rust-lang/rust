//@ run-pass


fn one() -> i32 { 1 }

// Make sure the vec![...] macro doesn't introduce hidden rvalue
// scopes (such as blocks) around the element expressions.
pub fn main() {
    assert_eq!(vec![&one(), &one(), &2], vec![&1, &1, &(one()+one())]);
    assert_eq!(vec![&one(); 2], vec![&1, &one()]);
}
