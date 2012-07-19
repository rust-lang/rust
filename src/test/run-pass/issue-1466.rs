// exec-env:RUST_CC_ZEAL=1
// xfail-test

fn main() {
    #error["%?", os::getenv(~"RUST_CC_ZEAL")];
    let _x = @{a: @10, b: ~true};
}
