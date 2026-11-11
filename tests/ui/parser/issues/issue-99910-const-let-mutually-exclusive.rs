//@ run-rustfix

fn main() {
    const let _FOO: i32 = 123;
    //~^ ERROR const` and `let` are mutually exclusive
    let const _BAR: i32 = 123;
    //~^ ERROR `const` and `let` are mutually exclusive
}
