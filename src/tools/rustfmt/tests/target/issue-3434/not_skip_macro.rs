#[this::is::not::skip::macros(ouch)]

fn main() {
    let macro_result1 = ouch! { <div>
    this should be mangled</div>
        }
    .to_string();
}
