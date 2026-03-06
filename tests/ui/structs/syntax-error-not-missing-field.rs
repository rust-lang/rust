// Check that a syntax error inside a struct literal does not also report missing fields,
// because the field might be present but hidden by the syntax error.
//
// The stderr for this test should contain ONLY one syntax error per struct literal,
// and not any errors about missing fields.
#![allow(todo_macro_uses)]

struct Foo { a: isize, b: isize }

enum Bar {
    Baz { a: isize, b: isize },
}

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

fn enum_expr_wrong_separator() {
    let e = Bar::Baz { a: make_a(); b: 2 }; //~ ERROR found `;`
}

fn enum_expr_missing_separator() {
    let e = Bar::Baz { a: make_a() b: 2 }; //~ ERROR found `b`
}

// Should error but not then ICE due to lack of type checking
fn regression_test_for_issue_153388_a() {
    struct TheStruct;
    struct MyStruct {
        value: i32,
        s: TheStruct,
    }
    static A: MyStruct = MyStruct { ,s: TheStruct }; //~ ERROR expected identifier, found `,`
}

fn regression_test_for_issue_153388_b() {
    struct MyStruct {
        value: i32,
    }
    static A: MyStruct = MyStruct {,}; //~ ERROR expected identifier, found `,`
}

fn main() {}
