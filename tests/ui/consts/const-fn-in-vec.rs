static _MAYBE_STRINGS: [Option<String>; 5] = [None; 5];
//~^ ERROR the trait bound `String: Copy` is not satisfied

// should hint to create an inline `const` block
// or to create a new `const` item
fn foo() {
    let _strings: [String; 5] = [String::new(); 5];
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn bar() {
    let _maybe_strings: [Option<String>; 5] = [None; 5];
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn main() {}
