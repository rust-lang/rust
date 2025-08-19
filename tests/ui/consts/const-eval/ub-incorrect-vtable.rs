// This test contains code with incorrect vtables in a const context:
// - from issue 86132: a trait object with invalid alignment caused an ICE in const eval, and now
//   triggers an error
// - a similar test that triggers a previously-untested const UB error: emitted close to the above
//   error, it checks the correctness of the size
//
// As is, this code will only hard error when the constants are used, and the errors are emitted via
// the `#[allow]`-able `const_err` lint. However, if the transparent wrapper technique to prevent
// reborrows is used -- from `ub-wide-ptr.rs` -- these two errors reach validation and would trigger
// ICEs as tracked by #86193. So we also use the transparent wrapper to verify proper validation
// errors are emitted instead of ICEs.

//@ stderr-per-bitwidth
//@ dont-require-annotations: NOTE

trait Trait {}

const INVALID_VTABLE_ALIGNMENT: &dyn Trait =
    unsafe { std::mem::transmute((&92u8, &[0usize, 1usize, 1000usize])) };
//~^^ ERROR vtable

const INVALID_VTABLE_SIZE: &dyn Trait =
    unsafe { std::mem::transmute((&92u8, &[1usize, usize::MAX, 1usize])) };
//~^^ ERROR vtable

#[repr(transparent)]
struct W<T>(T);

fn drop_me(_: *mut usize) {}

const INVALID_VTABLE_ALIGNMENT_UB: W<&dyn Trait> =
    unsafe { std::mem::transmute((&92u8, &(drop_me as fn(*mut usize), 1usize, 1000usize))) };
//~^^ ERROR expected a vtable pointer

const INVALID_VTABLE_SIZE_UB: W<&dyn Trait> =
    unsafe { std::mem::transmute((&92u8, &(drop_me as fn(*mut usize), usize::MAX, 1usize))) };
//~^^ ERROR expected a vtable pointer

// Even if the vtable has a fn ptr and a reasonable size+align, it still does not work.
const INVALID_VTABLE_UB: W<&dyn Trait> =
    unsafe { std::mem::transmute((&92u8, &(drop_me as fn(*mut usize), 1usize, 1usize))) };
//~^^ ERROR expected a vtable pointer

// Trying to access the data in a vtable does not work, either.

#[derive(Copy, Clone)]
struct Wide<'a>(&'a Foo, &'static VTable);

struct VTable {
    drop: Option<for<'a> fn(&'a mut Foo)>,
    size: usize,
    align: usize,
    bar: for<'a> fn(&'a Foo) -> u32,
}

trait Bar {
    fn bar(&self) -> u32;
}

struct Foo {
    foo: u32,
    bar: bool,
}

impl Bar for Foo {
    fn bar(&self) -> u32 {
        self.foo
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        assert!(!self.bar);
        self.bar = true;
        println!("dropping Foo");
    }
}

#[repr(C)]
union Transmute<T: Copy, U: Copy> {
    t: T,
    u: U,
}

const FOO: &dyn Bar = &Foo { foo: 128, bar: false };
const G: Wide = unsafe { Transmute { t: FOO }.u };
//~^ ERROR encountered a dangling reference
// (it is dangling because vtables do not contain memory that can be dereferenced)

fn main() {}
