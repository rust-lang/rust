//@compile-flags: -Zmiri-tree-borrows -Zmiri-tag-gc=0

#[path = "../../utils/mod.rs"]
mod utils;
use utils::macros::*;

// Check the formatting of the trees.
fn main() {
    unsafe {
        alignment_check();
        structure_check();
    }
}

// Alignment check: we split the array at indexes with different amounts of
// decimal digits to verify proper padding.
unsafe fn alignment_check() {
    let data: &mut [u8] = &mut [0; 1024];
    name!(data.as_ptr()=>2, "data");
    let alloc_id = alloc_id!(data.as_ptr());
    let x = &mut data[1];
    name!(x as *mut _, "data[1]");
    *x = 1;
    let x = &mut data[10];
    name!(x as *mut _, "data[10]");
    *x = 1;
    let x = &mut data[100];
    name!(x as *mut _, "data[100]");
    *x = 1;
    let _val = data[100]; // So that the above is Frz
    let x = &mut data[1000];
    name!(x as *mut _, "data[1000]");
    *x = 1;
    print_state!(alloc_id);
}

// Tree structure check: somewhat complex organization of reborrows.
unsafe fn structure_check() {
    let x = &0u8;
    name!(x);
    let xa = &*x;
    name!(xa);
    let xb = &*x;
    name!(xb);
    let xc = &*x;
    name!(xc);
    let xaa = &*xa;
    name!(xaa);
    let xab = &*xa;
    name!(xab);
    let xba = &*xb;
    name!(xba);
    let xbaa = &*xba;
    name!(xbaa);
    let xbaaa = &*xbaa;
    name!(xbaaa);
    let xbaaaa = &*xbaaa;
    name!(xbaaaa);
    let xca = &*xc;
    name!(xca);
    let xcb = &*xc;
    name!(xcb);
    let xcaa = &*xca;
    name!(xcaa);
    let xcab = &*xca;
    name!(xcab);
    let xcba = &*xcb;
    name!(xcba);
    let xcbb = &*xcb;
    name!(xcbb);
    let alloc_id = alloc_id!(x);
    print_state!(alloc_id);
}
