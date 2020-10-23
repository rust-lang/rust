// Regression test for #71732, it used to emit incorrect diagnostics, like:
// error[E0283]: type annotations needed
//  --> src/main.rs:5:10
//   |
// 5 |         .get(&"key".into())
//   |          ^^^ cannot infer type for struct `String`
//   |
//   = note: cannot satisfy `String: Borrow<_>`
// help: consider specifying the type argument in the method call
//   |
// 5 |         .get::<Q>(&"key".into())
//   |

use std::collections::hash_map::HashMap;

fn foo(parameters: &HashMap<String, String>) -> bool {
    parameters
        .get(&"key".into()) //~ ERROR: type annotations needed
        .and_then(|found: &String| Some(false))
        .unwrap_or(false)
}

fn main() {}
