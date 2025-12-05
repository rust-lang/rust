// skip-filecheck

// Validate that we record the target for the `as` coercion as `for<'a> fn(&'a (), &'a ())`,
// and not `for<'a, 'b>(&'a (), &'b ())`. We previously did the latter due to a bug in
// the code that records adjustments in HIR typeck.

fn foo<'a, 'b>(_: &'a (), _: &'b ()) {}

// EMIT_MIR build_correct_coerce.main.built.after.mir
fn main() {
    let x = foo as for<'a> fn(&'a (), &'a ());
}
