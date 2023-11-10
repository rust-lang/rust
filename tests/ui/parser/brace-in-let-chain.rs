// issue #117766

#![feature(let_chains)]
fn main() {
    if let () = ()
        && let () = () { //~ERROR: found a `{` in the middle of a let-chain
        && let () = ()
    {
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
