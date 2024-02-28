fn foo() -> String { String::new() }

fn main() {
    let string_arr = [foo(); 64]; //~ ERROR trait `Copy` is not implemented for `String`
}
