//@ run-pass
#![feature(fn_align)]

trait Test {
    #[align(4096)]
    fn foo(&self);

    #[align(4096)]
    fn foo1(&self);
}

fn main() {
    assert_eq!((<dyn Test>::foo as fn(_) as usize & !1) % 4096, 0);
    assert_eq!((<dyn Test>::foo1 as fn(_) as usize & !1) % 4096, 0);
}
