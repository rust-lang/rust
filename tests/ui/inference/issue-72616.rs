//@ ignore-wasm32 FIXME: ignoring wasm as it suggests slightly different impls

// Regression test for #72616, it used to emit incorrect diagnostics, like:
// error[E0283]: type annotations needed for `String`
//  --> src/main.rs:8:30
//   |
// 5 |         let _: String = "".to_owned().try_into().unwrap();
//   |             - consider giving this pattern a type
// ...
// 8 |         if String::from("a") == "a".try_into().unwrap() {}
//   |                              ^^ cannot infer type for struct `String`
//   |
//   = note: cannot satisfy `String: PartialEq<_>`

use std::convert::TryInto;

pub fn main() {
    {
        let _: String = "".to_owned().try_into().unwrap();
    }
    {
        if String::from("a") == "a".try_into().unwrap() {}
        //~^ ERROR type annotations needed
    }
    {
        let _: String = match "_".try_into() {
            Ok(a) => a,
            Err(_) => "".into(),
        };
    }
}
