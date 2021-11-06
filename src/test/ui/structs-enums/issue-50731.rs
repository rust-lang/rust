// run-pass
enum Void {}
fn foo(_: Result<(Void, u32), (Void, String)>) {}
fn main() {
    let _: fn(_) = foo;
}
