//@ known-bug: #138132
#![feature(min_generic_const_args)]
struct b(Box<[u8; c]>);
impl b {
    fn d(self) {
        self.0.e()
    }
}
struct c<'a>(&'a u8);
fn main() {}
