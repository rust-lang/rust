#![feature(cfg_select)]
#![crate_type = "lib"]

// Check that parse errors in arms that are not selected are still reported.

fn print() {
    println!(cfg_select! {
        false => { 1 ++ 2 }
        //~^ ERROR Rust has no postfix increment operator
        _ => { "not unix" }
    });
}

cfg_select! {
    false => { fn foo() { 1 +++ 2 } }
    //~^ ERROR Rust has no postfix increment operator
    _ => {}
}
