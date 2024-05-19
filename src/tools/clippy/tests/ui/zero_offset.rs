#[allow(clippy::borrow_as_ptr)]
fn main() {
    unsafe {
        let m = &mut () as *mut ();
        m.offset(0);
        //~^ ERROR: offset calculation on zero-sized value
        //~| NOTE: `#[deny(clippy::zst_offset)]` on by default
        m.wrapping_add(0);
        //~^ ERROR: offset calculation on zero-sized value
        m.sub(0);
        //~^ ERROR: offset calculation on zero-sized value
        m.wrapping_sub(0);
        //~^ ERROR: offset calculation on zero-sized value

        let c = &() as *const ();
        c.offset(0);
        //~^ ERROR: offset calculation on zero-sized value
        c.wrapping_add(0);
        //~^ ERROR: offset calculation on zero-sized value
        c.sub(0);
        //~^ ERROR: offset calculation on zero-sized value
        c.wrapping_sub(0);
        //~^ ERROR: offset calculation on zero-sized value

        let sized = &1 as *const i32;
        sized.offset(0);
    }
}
