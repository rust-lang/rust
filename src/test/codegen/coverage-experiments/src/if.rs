#![feature(core_intrinsics)]

pub fn __llvm_incr_counter(_region_loc: &str) {
}

#[inline(always)]
pub fn __incr_cov<T>(region_loc: &str, result: T) -> T {
    __llvm_incr_counter(region_loc);
    result
}

static TEST_FUNC_NAME: &'static [u8; 6] = b"main()";

fn main() {
    let mut countdown = 10;
    if __incr_cov("start", countdown > 0) {


        // // TEST CALLING INTRINSIC:
        unsafe { core::intrinsics::instrprof_increment(TEST_FUNC_NAME as *const u8, 1234 as u64, 314 as u32, 31 as u32) };
        // // Results in:
        // //   LLVM ERROR: Cannot select: intrinsic %llvm.instrprof.increment
        // // I may need to pass one or more of the following flags (or equivalent opts) to LLVM to enable this:
        // //   -fprofile-instr-generate -fcoverage-mapping


        countdown -= 1;
        __incr_cov("if block",());
    } else if countdown > 5 {
        countdown -= 2;
        __incr_cov("else if block",());
    } else {
        countdown -= 3;
    }

    let mut countdown = 10;
    if { let _tcov = countdown > 0; __llvm_incr_counter("start", ); _tcov } {
        countdown -= 1;
        __incr_cov("if block",());
    } else if countdown > 5 {
        countdown -= 2;
        __incr_cov("else if block",());
    } else {
        countdown -= 3;
    }
}

// NOTE: hir REDUNDANTLY lowers the manually inlined counter in the second if block to:
//
// match {
//   let _t =
//       {
//           let _tcov = countdown > 0;
//           __llvm_incr_counter("start");
//           _tcov
//       };
//   _t
// } {

// I don't know if optimization phases will fix this or not.
// Otherwise, a more optimal (but definitely special case) way to handle this would be
// to inject the counter between the hir-introduced temp `_t` assignment and the block result
// line returning `_t`:
//
// match {
//   let _t = countdown > 0;
//   __llvm_incr_counter("start"); // <-- the only thing inserted for coverage here
//   _t
// }
//
// UNFORTUNATELY THIS IS NOT A PATTERN WE CAN ALWAYS LEVERAGE, FOR EXPRESSIONS THAT HAVE VALUES
// WHERE WE NEED TO INJECT THE COUNTER AFTER THE EXPRESSION BUT BEFORE IT IS USED.
//
// IT DOES APPEAR TO BE THE CASE FOR WHILE EXPRESSIONS, (BECOMES loop { match { let _t = condition; _t} { true => {...} _ => break, }})
// AND IS TRUE FOR IF EXPRESSIONS AS NOTED
// BUT NOT FOR RETURN STATEMENT (and I'm guessing not for loop { break value; } ? )
//
// AND NOT FOR LAZY BOOLEAN EXPRESSIONS!
//
// AND NOT FOR MATCH EXPRESSIONS IN THE ORIGINAL SOURCE!