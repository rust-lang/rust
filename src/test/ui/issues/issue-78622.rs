struct S;
fn f() {
    S::A::<f> {} //~ ERROR ambiguous associated type
}
fn main() {}
