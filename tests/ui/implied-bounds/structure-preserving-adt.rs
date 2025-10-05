//@ check-pass
// Structure-preserving HRTB coercion with nested references inside an ADT.

struct Wrap<T>(T);

fn foo<'a, 'b>(_: Wrap<&'a &'b ()>, v: &'b u32) -> &'a u32 { v }

// Coerce to HRTB fn pointer that preserves nested structure via ADT.
fn takes_f(_: for<'x, 'y> fn(Wrap<&'x &'y ()>, &'y u32) -> &'x u32) {}

fn main() {
    takes_f(foo);
}
