extern crate ccc;

fn main() {
    ccc::do_work();
    ccc::do_work_generic::<i16>();
    ccc::do_work_generic::<i32>();
}
