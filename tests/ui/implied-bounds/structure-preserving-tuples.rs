//@ check-pass
// Structure-preserving HRTB coercion with nested references inside a 1-tuple.

static UNIT: &'static &'static () = &&();

fn foo<'a, 'b>(_: (&'a &'b (),), v: &'b u32) -> &'a u32 { v }

// Coerce to HRTB fn pointer that preserves nested structure.
fn takes_f(_: for<'x, 'y> fn((&'x &'y (),), &'y u32) -> &'x u32) {}

fn main() {
    takes_f(foo);
}
