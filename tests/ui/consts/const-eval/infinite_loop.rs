//! This test tests two things at once:
//! 1. we error if a const evaluation hits the deny-by-default lint limit
//! 2. we do not ICE on invalid follow-up code
//! 3. no ICE when run with `-Z unstable-options` (issue 122177)

//@revisions: eval_limit no_ice
//@[no_ice] compile-flags: -Z tiny-const-eval-limit -Z unstable-options
//@[eval_limit] compile-flags: -Z tiny-const-eval-limit

fn main() {
    // Tests the Collatz conjecture with an incorrect base case (0 instead of 1).
    // The value of `n` will loop indefinitely (4 - 2 - 1 - 4).
    let s = [(); {
        let mut n = 113383; // #20 in https://oeis.org/A006884
        while n != 0 {
            //~^ ERROR is taking a long time
            n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        }
        n
    }];

    s.nonexistent_method();
}
