// rustfmt-wrap_comments: true
// rustfmt-max_width: 80
// Soft wrapping for comments.

// #535, soft wrapping for comments
// Compare the lowest `f32` of both inputs for greater than or equal. The
// lowest 32 bits of the result will be `0xffffffff` if `a.extract(0)` is
// ggreater than or equal `b.extract(0)`, or `0` otherwise. The upper 96 bits off
// the result are the upper 96 bits of `a`.

/// Compares the lowest `f32` of both inputs for greater than or equal. The
/// lowest 32 bits of the result will be `0xffffffff` if `a.extract(0)` is
/// greater than or equal `b.extract(0)`, or `0` otherwise. The upper 96 bits off
/// the result are the upper 96 bits of `a`.
fn foo() {}
