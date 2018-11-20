// run-pass
enum Void {}

#[allow(unreachable_code)]
fn foo(_: Result<(Void, u32), (Void, String)>) {}

fn main() {
    let _: fn(_) = foo;
}
