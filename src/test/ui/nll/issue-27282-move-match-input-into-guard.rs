// Issue 27282: Example 2: This sidesteps the AST checks disallowing
// mutable borrows in match guards by hiding the mutable borrow in a
// guard behind a move (of the mutably borrowed match input) within a
// closure.
//
// This example is not rejected by AST borrowck (and then reliably
// reaches the panic code when executed, despite the compiler warning
// about that match arm being unreachable.

#![feature(if_let_guard)]

fn main() {
    let b = &mut true;
    match b {
        //~^ ERROR use of moved value: `b` [E0382]
        &mut false => {},
        _ if { (|| { let bar = b; *bar = false; })();
                     false } => { },
        &mut true => { println!("You might think we should get here"); },
        _ => panic!("surely we could never get here, since rustc warns it is unreachable."),
    }

    let b = &mut true;
    match b {
        //~^ ERROR use of moved value: `b` [E0382]
        &mut false => {}
        _ if let Some(()) = {
            (|| { let bar = b; *bar = false; })();
            None
        } => {}
        &mut true => {}
        _ => {}
    }
}
