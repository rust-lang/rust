// We disable the GC for this test because it would change what is printed.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance -Zmiri-provenance-gc=0

#[path = "../../../utils/mod.rs"]
#[macro_use]
mod utils;

fn main() {
    unsafe {
        let x = &0u8;
        name!(x);
        let xa = &*x;
        name!(xa);
        let xb = &*x;
        name!(xb);
        let wild = xb as *const u8 as usize as *const u8;

        let y = &*wild;
        name!(y);
        let ya = &*y;
        name!(ya);
        let yb = &*y;
        name!(yb);
        let _int = ya as *const u8 as usize;

        let z = &*wild;
        name!(z);

        let u = &*wild;
        name!(u);
        let ua = &*u;
        name!(ua);
        let alloc_id = alloc_id!(x);
        print_state!(alloc_id);
    }
}
