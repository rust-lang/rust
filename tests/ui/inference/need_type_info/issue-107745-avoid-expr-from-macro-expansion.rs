// ignore-tidy-linelength

// Regression test for #107745.
// Previously need_type_info::update_infer_source will consider expressions originating from
// macro expressions as candiate "previous sources". This unfortunately can mean that
// for macros expansions such as `format!()` internal implementation details can leak, such as:
//
// ```
// error[E0282]: type annotations needed
// --> src/main.rs:2:22
//  |
//2 |     println!("{:?}", []);
//  |                      ^^ cannot infer type of the type parameter `T` declared on the associated function `new_debug`
// ```

fn main() {
    println!("{:?}", []);
    //~^ ERROR type annotations needed
}
