// This test captures a review comment example where nested references appear
// in the RETURN TYPE and a chain of HRTB fn-pointer coercions allows
// unsound lifetime extension. This should be rejected, but currently passes.
// We annotate the expected error to reveal the gap.

fn foo<'out, 'input, T>(_dummy: &'out (), value: &'input T) -> (&'out &'input (), &'out T) {
    (&&(), value)
}

fn bad<'short, T>(x: &'short T) -> &'static T {
    let foo1: for<'out, 'input> fn(&'out (), &'input T) -> (&'out &'input (), &'out T) = foo;
    let foo2: for<'input> fn(&'static (), &'input T) -> (&'static &'input (), &'static T) = foo1;
    let foo3: for<'input> fn(&'static (), &'input T) -> (&'input &'input (), &'static T) = foo2; //~ ERROR mismatched types
    let foo4: fn(&'static (), &'short T) -> (&'short &'short (), &'static T) = foo3;
    foo4(&(), x).1
}

fn main() {
    let s = String::from("hi");
    let _r: &'static String = bad(&s);
}
