// run-pass
// rustc_env: overridden=no
// compile-flags: -Z unstable-options --env x=y --env overridden=yes

fn main() {
    assert_eq!(env!("x"), "y");
    assert_eq!(env!("overridden"), "yes");
}
