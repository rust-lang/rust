#![rustfmt::skip::macros(skip_macro_mod)]

mod no_entry;

#[rustfmt::skip::macros(html, skip_macro)]
fn main() {
    let macro_result1 = html! { <div>
this should be skipped</div>
    }
    .to_string();

    let macro_result2 = not_skip_macro! { <div>
this should be mangled</div>
        }
    .to_string();

    skip_macro! {
this should be skipped
};

    foo();
}

fn foo() {
    let macro_result1 = html! { <div>
this should be mangled</div>
            }
    .to_string();
}

fn bar() {
    let macro_result1 = skip_macro_mod! { <div>
this should be skipped</div>
        }
    .to_string();
}

fn visitor_made_from_same_context() {
    let pair = (
        || {
            foo!(<div>
this should be mangled</div>
            );
            skip_macro_mod!(<div>
this should be skipped</div>
            );
        },
        || {
            foo!(<div>
this should be mangled</div>
            );
            skip_macro_mod!(<div>
this should be skipped</div>
            );
        },
    );
}
