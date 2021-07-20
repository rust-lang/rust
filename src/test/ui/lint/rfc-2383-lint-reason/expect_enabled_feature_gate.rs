// check-pass

#![feature(lint_reasons)]

// should be fine due to the enabled feature gate
#[expect(unused_variables)]
fn main() {
    let x = 1;
}
