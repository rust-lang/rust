// This function is to check whether `if-let` guard error points to
// `let_chains` feature gate, which is wrong and confuse people.
pub fn foo(a: &[u8], b: bool) -> bool {
    match b {
        true if let [1, 2, 3, ..] = a => true,
        //~^    ERROR `let` expressions are not supported here
        //~^^   NOTE only supported directly in conditions of `if`- and `while`-expressions
        //~^^^  NOTE as well as when nested within `&&` and parenthesis in those conditions
        _ => false,
    }
}

fn main() {}
