fn foo() -> String { String::new() }

fn main() {
    let string_arr = [foo(); 64]; //~ ERROR the trait bound `String: Copy` is not satisfied
}
