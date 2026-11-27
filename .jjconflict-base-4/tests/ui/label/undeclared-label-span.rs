//! regression test for <https://github.com/rust-lang/rust/issues/28109>
//! Make sure that label for continue and break is spanned correctly.

fn main() {
    loop {
        continue
        'b //~ ERROR use of undeclared label
        ;
        break
        'c //~ ERROR use of undeclared label
        ;
    }
}
