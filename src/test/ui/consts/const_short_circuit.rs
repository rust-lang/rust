const _: bool = false && false;
const _: bool = true && false;
const _: bool = {
    let mut x = true && false;
    //~^ ERROR new features like let bindings are not permitted
    x
};
const _: bool = {
    let x = true && false;
    //~^ ERROR new features like let bindings are not permitted
    x
};

fn main() {}
