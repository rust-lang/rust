// skip-filecheck
//@ test-mir-pass: AddRetag
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// ignore-tidy-linelength
//@ compile-flags: -Z mir-emit-retag -Z mir-opt-level=0 -Z span_free_formats

#![allow(unused)]

struct Test(i32);

// EMIT_MIR retag.{impl#0}-foo.SimplifyCfg-pre-optimizations.after.mir
// EMIT_MIR retag.{impl#0}-foo_shr.SimplifyCfg-pre-optimizations.after.mir
impl Test {
    // Make sure we run the pass on a method, not just on bare functions.
    fn foo<'x>(&self, x: &'x mut i32) -> &'x mut i32 {
        x
    }
    fn foo_shr<'x>(&self, x: &'x i32) -> &'x i32 {
        x
    }
}

// EMIT_MIR core.ptr-drop_in_place.Test.SimplifyCfg-make_shim.after.mir

impl Drop for Test {
    fn drop(&mut self) {}
}

// EMIT_MIR retag.main.SimplifyCfg-pre-optimizations.after.mir
// EMIT_MIR retag.main-{closure#0}.SimplifyCfg-pre-optimizations.after.mir
pub fn main() {
    let mut x = 0;
    {
        let v = Test(0).foo(&mut x); // just making sure we do not panic when there is a tuple struct ctor
        let w = { v }; // assignment
        let w = w; // reborrow
        // escape-to-raw (mut)
        let _w = w as *mut _;
    }

    // Also test closures
    let c: fn(&i32) -> &i32 = |x: &i32| -> &i32 {
        let _y = x;
        x
    };
    let _w = c(&x);

    // need to call `foo_shr` or it doesn't even get generated
    Test(0).foo_shr(&0);

    // escape-to-raw (shr)
    let _w = _w as *const _;

    array_casts();
}

/// Casting directly to an array should also go through `&raw` and thus add appropriate retags.
// EMIT_MIR retag.array_casts.SimplifyCfg-pre-optimizations.after.mir
fn array_casts() {
    let mut x: [usize; 2] = [0, 0];
    let p = &mut x as *mut usize;
    unsafe {
        *p.add(1) = 1;
    }

    let x: [usize; 2] = [0, 1];
    let p = &x as *const usize;
    assert_eq!(unsafe { *p.add(1) }, 1);
}

// EMIT_MIR retag.box_to_raw_mut.SimplifyCfg-pre-optimizations.after.mir
fn box_to_raw_mut(x: &mut Box<i32>) -> *mut i32 {
    std::ptr::addr_of_mut!(**x)
}
