// check-pass
// known-bug: #25860

// Should fail. The combination of variance and implied bounds for nested
// references allows us to infer a longer lifetime than we can prove.

static UNIT: &'static &'static () = &&();

fn foo<'a, 'b, T>(_: &'a &'b (), v: &'b T) -> &'a T { v }

fn bad<'a, T>(x: &'a T) -> &'static T {
    let f: fn(_, &'a T) -> &'static T = foo;
    f(UNIT, x)
}

fn main() {}
