fn a() {
    let Some(x) = foo() { //~ ERROR expected one of
        //~^ HELP you might have meant to use `if let`
        let y = x;
    }
}
fn b() {
    let Some(x) = foo() { //~ ERROR expected one of
        //~^ HELP you might have meant to use `let else`
        return;
    }
}
fn c() {
    let Some(x) = foo() { //~ ERROR expected one of
        //~^ HELP you might have meant to use `if let`
        //~| HELP alternatively, you might have meant to use `let else`
        // The parser check happens pre-macro-expansion, so we don't know for sure.
        println!("{x}");
    }
}
fn foo() -> Option<i32> {
    Some(42)
}
fn main() {}
