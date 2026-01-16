// Auxiliary crate for testing -Z separate-spans flag.
// When separate-spans is enabled, span data is stored in a separate .spans file
// instead of being embedded directly in the .rmeta file.

//@[rpass1] compile-flags: -Z query-dep-graph
//@[rpass2] compile-flags: -Z query-dep-graph
//@[rpass3] compile-flags: -Z query-dep-graph -Z separate-spans

#![crate_type = "rlib"]

#[inline(always)]
pub fn inlined_with_span() -> u32 {
    let x = 1;
    let y = 2;
    x + y
}

#[inline(always)]
pub fn generic_with_span<T: std::fmt::Debug>(val: T) -> String {
    format!("{:?}", val)
}

pub fn multi_span_fn() -> (u32, u32, u32) {
    let a = compute_a();
    let b = compute_b();
    let c = compute_c();
    (a, b, c)
}

fn compute_a() -> u32 { 1 }
fn compute_b() -> u32 { 2 }
fn compute_c() -> u32 { 3 }

/// Macro that generates code with spans.
#[macro_export]
macro_rules! generate_fn {
    ($name:ident, $val:expr) => {
        pub fn $name() -> u32 {
            $val
        }
    };
}
