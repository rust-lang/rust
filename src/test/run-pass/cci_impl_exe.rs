// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_impl_lib.rs

use cci_impl_lib;
import cci_impl_lib::helpers;
import cci_impl_lib::uint_helpers;

fn main() {
    //let bt0 = sys::frame_address();
    //#debug["%?", bt0];

    do 3u.to(10u) |i| {
        io::print(#fmt["%u\n", i]);

        //let bt1 = sys::frame_address();
        //#debug["%?", bt1];
        //assert bt0 == bt1;
    }
}
