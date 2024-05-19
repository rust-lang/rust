#[rustfmt::skip::macros(another_macro)]
fn foo() {
    another_macro!(
This should be skipped.
        );
}

fn bar() {
    skip_macro_mod!(
This should be skipped.
        );
}

fn baz() {
    let macro_result1 = no_skip_macro! { <div>
this should be mangled</div>
    }.to_string();
}
