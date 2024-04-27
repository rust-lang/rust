//@ check-pass

const FOO: isize = 10;
const ZST: &() = unsafe { std::mem::transmute(FOO) };
fn main() {
    match &() {
        ZST => 9,
    };
}
