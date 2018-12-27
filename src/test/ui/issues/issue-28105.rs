// Make sure that a continue span actually contains the keyword.

fn main() {
    continue //~ ERROR `continue` outside of loop
    ;
    break //~ ERROR `break` outside of loop
    ;
}
