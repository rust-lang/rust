// Check that a syntax error inside a struct literal does not also report missing fields,
// because the field might be present but hidden by the syntax error.
//
// The stderr for this test should contain ONLY one syntax error per struct literal,
// and not any errors about missing fields.

struct Foo { a: isize, b: isize }

fn make_a() -> isize { 1234 }

fn expr_wrong_separator() {
    let f = Foo { a: make_a(); b: 2 }; //~ ERROR found `;`
}

fn expr_missing_separator() {
    let f = Foo { a: make_a() b: 2 }; //~ ERROR found `b`
}

fn expr_rest_trailing_comma() {
    let f = Foo { a: make_a(), ..todo!(), }; //~ ERROR cannot use a comma
}

fn expr_missing_field_name() {
    let f = Foo { make_a(), b: 2, }; //~ ERROR found `(`
}

fn pat_wrong_separator(Foo { a; b }: Foo) { //~ ERROR expected `,`
    let _ = (a, b);
}

fn pat_missing_separator(Foo { a b }: Foo) { //~ ERROR expected `,`
    let _ = (a, b);
}

fn pat_rest_trailing_comma(Foo { a, .., }: Foo) { //~ ERROR expected `}`, found `,`
}

fn main() {}
