fn main() {
    transmute(); // does not ICE
    //~^ ERROR call to unsafe function `transmute` is unsafe and requires unsafe function or block
}

#[rustc_intrinsic]
//~^ ERROR the `#[rustc_intrinsic]` attribute is used to declare intrinsics as function items
unsafe fn transmute() {}
