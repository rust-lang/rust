//@ run-pass
#[derive(Eq, PartialEq)]
pub struct Data([u8; 4]);

const DATA: Data = Data([1, 2, 3, 4]);

fn main() {
    match DATA {
        DATA => (),
        _ => (),
    }
}
