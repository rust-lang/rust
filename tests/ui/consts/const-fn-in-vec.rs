static _MAYBE_STRINGS: [Option<String>; 5] = [None; 5];
//~^ ERROR the trait bound `String: Copy` is not satisfied

fn main() {
    // should hint to create an inline `const` block
    // or to create a new `const` item
    let _strings: [String; 5] = [String::new(); 5];
    //~^ ERROR the trait bound `String: Copy` is not satisfied
    let _maybe_strings: [Option<String>; 5] = [None; 5];
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}
