fn main() {
    { unsafe 92 } //~ ERROR expected `{`, found `92`
}

fn foo() {
    { mod 92 } //~ ERROR expected identifier, found `92`
}

fn bar() {
    { trait 92 } //~ ERROR expected identifier, found `92`
}
