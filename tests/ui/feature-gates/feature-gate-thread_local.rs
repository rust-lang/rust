// Test that `#[thread_local]` attribute is gated by `thread_local`
// feature gate.
//
// (Note that the `thread_local!` macro is explicitly *not* gated; it
// is given permission to expand into this unstable attribute even
// when the surrounding context does not have permission to use it.)

#[thread_local] //~ ERROR `#[thread_local]` is an experimental feature
static FOO: i32 = 3;

pub fn main() {}
