#![feature(stmt_expr_attributes)]

fn main() {
    // fold_stmt (Item)
    #[allow(dead_code)]
    #[derive(Debug)] // should not warn
    struct Foo;

    // fold_stmt (Mac)
    #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    println!("Hello, world!");

    // fold_stmt (Semi)
    #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    "Hello, world!";

    // fold_stmt (Local)
    #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    let _ = "Hello, world!";

    // visit_expr
    let _ = #[derive(Debug)] "Hello, world!";
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s

    let _ = [
        // filter_map_expr
        #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
        "Hello, world!",
    ];
}
