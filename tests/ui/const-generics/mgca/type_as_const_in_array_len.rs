//! Regression test for <https://github.com/rust-lang/rust/issues/138132>

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

struct B(Box<[u8; C]>);
//~^ ERROR missing generics for struct `C`
impl B {
    fn d(self) {
        self.0.e()
    }
}
struct C<'a>(&'a u8);
fn main() {}
