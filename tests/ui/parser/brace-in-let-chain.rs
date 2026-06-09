// issue #117766
//@ edition: 2024

fn main() {
    if let () = ()
        && let () = () {
        && let () = ()
    {
    }
}

fn quux() {
    while let () = ()
        && let () = () {
        && let () = ()
    {
    }
}

fn foobar() {
    while false {}
    {
        && let () = ()
}

fn fubar() {
    while false {
        {
            && let () = ()
    }
}

fn qux() {
    let foo = false;
    match foo {
        _ if foo => {
            && let () = ()
        _ => {}
    }
}

fn foo() {
    {
    && let () = ()
}

fn bar() {
    if false {}
    {
        && let () = ()
}

fn baz() {
    if false {
        {
            && let () = ()
    }
} //~ERROR: this file contains an unclosed delimiter
