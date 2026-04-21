#![feature(diagnostic_on_type_error)]
#[diagnostic::on_type_error(note = "expected struct `{Expected}`\n found struct `{Found}`")]
struct S<T>(T);
struct K<T> {
    foo: T,
}
fn main() {
    let s: S<i32> = S(String::new());
    //~^ ERROR mismatched types
    let k: K<i32> = K { foo: "" };
    //~^ ERROR mismatched types
    let _: S<i32> = k;
    //~^ ERROR mismatched types
}
