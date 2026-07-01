//@ known-bug: rust-lang/rust#143094
fn main() {
    #[cold]
    5
}

fn TokenStream() {
    #[rustc_align(16)]
    1u32
}

fn a() {
    #[macro_use]
    1
}

fn b() {
    #[loop_match]
    5
}

pub fn c() {
    #[crate_name = "xcrate_issue_61711_b"]
    0
}

pub fn d() {
    fn k() {}
    #[inline(always)]
    || -> fn() { k }
}

fn e() {
    #[inline]
    0
}

pub fn f() {
    #[proc_macro_derive(Bleh)]
    0
}
// etc..
