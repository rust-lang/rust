// issue: rust-lang/rust#105047
// ICE raw ptr comparison should already be caught in the trait systems

#![feature(raw_ref_op)]

const RCZ: *const i32 = &raw const *&0;

const fn f() {
    if let RCZ = &raw const *&0 { }
    //~^ WARN function pointers and raw pointers not derived from integers in patterns
    //~| ERROR pointers cannot be reliably compared during const eval
    //~| WARN this was previously accepted by the compiler but is being phased out
}

fn main() {}
