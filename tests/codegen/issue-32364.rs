// Test that `extern "stdcall"` is properly translated.

// only-x86

// compile-flags: -C no-prepopulate-passes

struct Foo;

impl Foo {
// CHECK: define internal x86_stdcallcc void @{{.*}}foo{{.*}}()
    #[inline(never)]
    pub extern "stdcall" fn foo<T>() {
    }
}

fn main() {
    Foo::foo::<Foo>();
}
