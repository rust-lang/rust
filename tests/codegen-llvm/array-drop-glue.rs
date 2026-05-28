//@ revisions: RAW OPT
//@ compile-flags: -C opt-level=z -C panic=abort
//@[RAW] compile-flags: -C no-prepopulate-passes -Z inline-mir

#![crate_type = "lib"]

// Ensure all the different array drop_glue functions just delegate to the slice one,
// rather than emitting two loops in each of the three.

// When this test was first written, the array drop glues came out in the
// seemingly-arbitrary order of 42, then 7, then 13, so to avoid potential
// fragility from that changing we don't check any particular order.

// RAW: ; core::ptr::drop_glue::<[array_drop_glue::NeedsDrop; [[N:7|13|42]]]>
// RAW-NEXT: inlinehint
// RAW: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
// RAW-NEXT: noundef [[N]])
// RAW: }

// RAW: ; core::ptr::drop_glue::<[array_drop_glue::NeedsDrop; [[N:7|13|42]]]>
// RAW-NEXT: inlinehint
// RAW: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
// RAW-NEXT: noundef [[N]])
// RAW: }

// RAW: ; core::ptr::drop_glue::<[array_drop_glue::NeedsDrop; [[N:7|13|42]]]>
// RAW-NEXT: inlinehint
// RAW: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
// RAW-NEXT: noundef [[N]])
// RAW: }

// CHECK-LABEL: ; core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
// CHECK-NOT: inlinehint
// OPT: add nuw nsw {{.+}} 1
// CHECK: }

#[no_mangle]
// CHECK-LABEL: @drop_arrays
pub fn drop_arrays(x: [NeedsDrop; 7], y: [NeedsDrop; 13], z: [NeedsDrop; 42]) {
    // I don't remember the parameter drop order, so write out the order the test expects.

    // RAW: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop; 7]>
    // OPT: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
    // OPT-NEXT: noundef 7)
    drop(x);
    // RAW: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop; 13]>
    // OPT: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
    // OPT-NEXT: noundef 13)
    drop(y);
    // RAW: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop; 42]>
    // OPT: call core::ptr::drop_glue::<[array_drop_glue::NeedsDrop]>
    // OPT-NEXT: noundef 42)
    drop(z);
}

struct NeedsDrop(u32);

impl Drop for NeedsDrop {
    #[inline]
    fn drop(&mut self) {
        do_the_drop(self);
    }
}

unsafe extern "Rust" {
    safe fn do_the_drop(_: &mut NeedsDrop);
}
