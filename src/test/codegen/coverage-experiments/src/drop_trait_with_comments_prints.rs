//
//
//
// It's interesting to speculate if there is a way to leverage the Drop trait functionality
// to increment counters when a scope is closed, but I don't think it would help "out of the box".
//
// A `return` or `break` with expression might not need a temp value expression wrapper
// such as `return { let _t = result_expression; __incr_counter(...); _t };`
//
//    ... **if** the __incr_counter() was somehow called from a "drop()" trait function.
//
// The problem is, since the drop call is automatic, there is no way to have argument variants
// depending on where the drop() occurs (e.g., from a `return` statement vs. from the end of
// the function). We need 2 different code regions though.
//
//
//
//

#[inline(always)]
pub fn __incr_cov<T>(_region_loc: &str, /*index: u32,*/ result: T) -> T {
    // println!("from: {}", _region_loc);
    result
}

struct Firework {
    strength: i32,
}

impl Drop for Firework {
    fn drop(&mut self) {
        println!("BOOM times {}!!!", self.strength);
        __incr_cov("start of drop()", ());
    }
}

fn main() -> Result<(),u8> {
    let _firecracker = Firework { strength: 1 };

    if __incr_cov("start of main()", true) {
        return __incr_cov("if true", { let _t = Err(1); println!("computing return value"); _t });
    }

    let _tnt = Firework { strength: 100 };
    // __incr_cov("after if block", Ok(())) // CAN USE COUNTER EXPRESSION: "start of drop()" - "if true"
    Ok(())
}

// OUTPUT WHEN RUNNING THIS PROGRAM IS AS EXPECTED:

// computing return value
// BOOM times 1!!!
// Error: 1
