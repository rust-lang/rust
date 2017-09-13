#![allow(dead_code)]

#[derive(Debug)]
struct A;

fn main() {
    // can't use assert_eq, b/c that will try to print the pointer addresses with full MIR enabled

    // FIXME: Test disabled for now, see <https://github.com/solson/miri/issues/131>.
    //assert!(&A as *const A as *const () == &() as *const _);
    //assert!(&A as *const A == &A as *const A);
}
