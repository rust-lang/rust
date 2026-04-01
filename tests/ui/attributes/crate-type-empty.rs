// regression test for issue 11256
#![crate_type]  //~ ERROR malformed `crate_type` attribute

fn main() {
    return
}
