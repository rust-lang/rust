#[rustfmt::skip::macros(html, skip_macro)]
fn main() {
    let macro_result1 = html! { <div>
Hello</div>
    }.to_string();

    let macro_result2 = not_skip_macro! { <div>
Hello</div>
    }.to_string();

    skip_macro! {
this is a skip_macro here
};

  foo();
}

fn foo() {
    let macro_result1 = html! { <div>
Hello</div>
    }.to_string();
}
