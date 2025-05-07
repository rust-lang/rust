//@ run-pass
//
// issue: <https://github.com/rust-lang/rust/issues/131813>

trait Pollable {
    #[allow(unused)]
    fn poll(&self) {}
}
trait FileIo: Pollable + Send + Sync {
    fn read(&self) {}
}
trait Terminal: Send + Sync + FileIo {}

struct A;

impl Pollable for A {}
impl FileIo for A {}
impl Terminal for A {}

fn main() {
    let a = A;

    let b = &a as &dyn Terminal;
    let c = b as &dyn FileIo;

    c.read();
}
