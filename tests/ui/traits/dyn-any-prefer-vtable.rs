//@ run-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn main() {
    let x: &dyn std::any::Any = &1i32;
    assert_eq!(x.type_id(), std::any::TypeId::of::<i32>());
}
