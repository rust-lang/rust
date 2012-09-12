// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_iter_lib.rs

extern mod cci_iter_lib;

fn main() {
    //let bt0 = sys::rusti::frame_address(1u32);
    //debug!("%?", bt0);
    do cci_iter_lib::iter(~[1, 2, 3]) |i| {
        io::print(fmt!("%d", i));
        //assert bt0 == sys::rusti::frame_address(2u32);
    }
}
