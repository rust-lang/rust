//@ known-bug: #138265

#![feature(coerce_unsized)]
#![crate_type = "lib"]
impl<A> std::ops::CoerceUnsized<A> for A {}
pub fn f() {
    [0; {
        let mut c = &0;
        c = &0;
        0
    }]
}
