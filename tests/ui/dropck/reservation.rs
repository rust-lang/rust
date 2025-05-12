#![feature(rustc_attrs)]

struct ReservedDrop;
#[rustc_reservation_impl = "message"]
impl Drop for ReservedDrop {
//~^ ERROR reservation `Drop` impls are not supported
    fn drop(&mut self) {}
}

fn main() {}
