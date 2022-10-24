// compile-flags: -Z track-diagnostics
// error-pattern: created at

struct A;
struct B;
const S: A = B;

fn main() {}
