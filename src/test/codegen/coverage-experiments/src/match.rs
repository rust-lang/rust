pub fn __llvm_incr_counter(_region_loc: &str) {
}

#[inline(always)]
pub fn __incr_cov<T>(region_loc: &str, result: T) -> T {
    __llvm_incr_counter(region_loc);
    result
}

fn main() {
    let a = 1;
    let b = 10;
    let _result = match a < b {
        true => true,
        _ => false,
    };

    let _result = match __incr_cov("end of first match", a < b) {
        true => __incr_cov("matched true", true),
        _ => false, // counter expression "end of first match" - "matched true"
    };
}