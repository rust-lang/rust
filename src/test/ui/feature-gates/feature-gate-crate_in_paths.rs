struct S;

fn main() {
    let _ = crate::S; //~ ERROR `crate` in paths is experimental
}
