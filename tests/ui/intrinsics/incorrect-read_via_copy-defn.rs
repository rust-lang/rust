fn main() {
    read_via_copy();
    //~^ ERROR call to unsafe function `read_via_copy` is unsafe and requires unsafe function or block
}

#[rustc_intrinsic]
//~^ ERROR the `#[rustc_intrinsic]` attribute is used to declare intrinsics as function items
unsafe fn read_via_copy() {}
