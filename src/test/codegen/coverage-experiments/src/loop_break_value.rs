pub fn __llvm_incr_counter(_region_loc: &str) {
}

#[inline(always)]
pub fn __incr_cov<T>(region_loc: &str, result: T) -> T {
    __llvm_incr_counter(region_loc);
    result
}

fn main() {
    __incr_cov("start", ());
    let _result = loop {
        break __incr_cov("top of loop", true);
    };
}