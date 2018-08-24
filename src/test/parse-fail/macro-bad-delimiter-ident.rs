// compile-flags: -Z parse-only

fn main() {
    foo! bar < //~ ERROR expected `(` or `{`, found `<`
}
