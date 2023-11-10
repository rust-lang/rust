// Issue #117720

#![feature(let_chains)]

fn main() {
    if let () = ()
        && let () = (); //~ERROR
        && let () = ()
    {
    }
}

fn foo() {
    if let () = ()
        && () == (); //~ERROR
        && 1 < 0
    {
    }
}

fn bar() {
    if let () = ()
        && () == (); //~ERROR
        && let () = ()
    {
    }
}
