#[allow(clippy::borrow_as_ptr, clippy::ptr_offset_by_literal)]
fn main() {
    unsafe {
        let m = &mut () as *mut ();
        m.offset(0);
        //~^ zst_offset

        m.wrapping_add(0);
        //~^ zst_offset

        m.sub(0);
        //~^ zst_offset

        m.wrapping_sub(0);
        //~^ zst_offset

        let c = &() as *const ();
        c.offset(0);
        //~^ zst_offset

        c.wrapping_add(0);
        //~^ zst_offset

        c.sub(0);
        //~^ zst_offset

        c.wrapping_sub(0);
        //~^ zst_offset

        let sized = &1 as *const i32;
        sized.offset(0);

        let nn = core::ptr::NonNull::<()>::dangling();
        nn.add(0);
        //~^ zst_offset

        nn.offset(0);
        //~^ zst_offset

        nn.sub(0);
        //~^ zst_offset

        let nn_sized = core::ptr::NonNull::<i32>::dangling();
        nn_sized.add(0);
    }
}
