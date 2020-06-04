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
    let c = 100;
    let _result = __incr_cov("start", a < b) || __incr_cov("or", b < c);

    let _result = { let _t = a < b; __llvm_incr_counter("start"); _t } || { let _t = b < c; __llvm_incr_counter("start"); _t };
}