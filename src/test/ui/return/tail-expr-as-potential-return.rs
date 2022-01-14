// > Suggest `return`ing tail expressions that match return type
// >
// > Some newcomers are confused by the behavior of tail expressions,
// > interpreting that "leaving out the `;` makes it the return value".
// > To help them go in the right direction, suggest using `return` instead
// > when applicable.
// (original commit description for this test)
//
// This test was amended to also serve as a regression test for #92308, where
// this suggestion would not trigger with async functions.
//
// edition:2018

fn main() {
    let _ = foo(true);
}

fn foo(x: bool) -> Result<f64, i32> {
    if x {
        Err(42) //~ ERROR mismatched types
                //| HELP you might have meant to return this value
    }
    Ok(42.0)
}

async fn bar(x: bool) -> Result<f64, i32> {
    if x {
        Err(42) //~ ERROR mismatched types
                //| HELP you might have meant to return this value
    }
    Ok(42.0)
}
