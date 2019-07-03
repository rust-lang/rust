// build-pass (FIXME(62277): could be check-pass?)

#![feature(stmt_expr_attributes)]
#![warn(unused_attributes)] //~ NOTE lint level defined here

fn foo<#[derive(Debug)] T>() { //~ WARN unused attribute
    match 0 {
        #[derive(Debug)] //~ WARN unused attribute
        _ => (),
    }
}

fn main() {
    // fold_stmt (Item)
    #[allow(dead_code)]
    #[derive(Debug)] // should not warn
    struct Foo;

    // fold_stmt (Mac)
    #[derive(Debug)]
    //~^ WARN `#[derive]` does nothing on macro invocations
    //~| NOTE this may become a hard error in a future release
    println!("Hello, world!");

    // fold_stmt (Semi)
    #[derive(Debug)] //~ WARN unused attribute
    "Hello, world!";

    // fold_stmt (Local)
    #[derive(Debug)] //~ WARN unused attribute
    let _ = "Hello, world!";

    // visit_expr
    let _ = #[derive(Debug)] "Hello, world!";
    //~^ WARN unused attribute

    let _ = [
        // filter_map_expr
        #[derive(Debug)] //~ WARN unused attribute
        "Hello, world!"
    ];
}
