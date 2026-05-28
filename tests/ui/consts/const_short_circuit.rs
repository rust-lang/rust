//@ check-pass

const _: bool = false && false;
const _: bool = true && false;
const _: bool = {
    let mut x = true && false;
    x
};
const _: bool = {
    let x = true && false;
    x
};

fn main() {}
