// Make sure that a continue span actually contains the keyword.

fn main() {
    continue //~ ERROR `continue` outside of a loop
    ;
    break //~ ERROR `break` outside of a loop
    ;
}
