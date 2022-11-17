// Tests that the compiler can mark `drop_in_place` as `noalias` when safe to do so.

#![crate_type="lib"]

use std::hint::black_box;

// CHECK: define{{.*}}core{{.*}}ptr{{.*}}drop_in_place{{.*}}Foo{{.*}}({{.*}}noalias {{.*}} align 4 dereferenceable(12){{.*}})

#[repr(C)]
pub struct Foo {
    a: i32,
    b: i32,
    c: i32,
}

impl Drop for Foo {
    #[inline(never)]
    fn drop(&mut self) {
        black_box(self.a);
    }
}

extern {
    fn bar();
    fn baz(foo: Foo);
}

pub fn haha() {
    let foo = Foo { a: 1, b: 2, c: 3 };
    unsafe {
        bar();
        baz(foo);
    }
}
