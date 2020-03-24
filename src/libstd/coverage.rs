//! Code coverage counters and report
//!
//! The code coverage library is typically not included in Rust source files.
//!
//! Instead, the `rustc` compiler optionally injects coverage calls into the internal representation
//! of the source if requested by command line option to the compiler. The end result is an
//! executable that includes the coverage counters, and writes a coverage report upon exit.
//!
//! The injected calls behave as if code like the following examples was actually part of the
//! original source.
//!
//! Example:
//!
//! ```
//! main() {
//!     let value = if (true) {
//!         std::coverage::count(1, {
//!             // any expression
//!             1000
//!         })
//!     } else {
//!         std::coverage::count(2, 500)
//!     }
//!     std::coverage::count_and_report(3, ())
//! }
//! ```

#![stable(feature = "coverage", since = "1.44.0")]

use crate::collections::HashMap;
use crate::sync::LockResult;
use crate::sync::Mutex;
use crate::sync::MutexGuard;
use crate::sync::Once;

static mut COUNTERS: Option<Mutex<HashMap<u128, usize>>> = None;

static INIT: Once = Once::new();

#[stable(feature = "coverage", since = "1.44.0")]
#[inline]
fn increment_counter(counter: u128) -> LockResult<MutexGuard<'static, HashMap<u128, usize>>> {
    let mut lock = unsafe {
        INIT.call_once(|| {
            COUNTERS = Some(Mutex::new(HashMap::with_capacity(1024)));
        });
        COUNTERS.as_mut().unwrap().lock()
    };
    let counters = lock.as_mut().unwrap();
    match counters.get_mut(&counter) {
        Some(count) => *count += 1,
        None => { counters.insert(counter,1); }
    }
    lock
}

/// The statement may be the last statement of a block, the value of a `return` or `break`,
/// a match pattern arm statement that might not be in a block (but if it is, don't wrap both the
/// block and the last statement of the block), or a closure statement without braces.
///
///     coverage::count(234234, {some_statement_with_or_without_semicolon()})
#[stable(feature = "coverage", since = "1.44.0")]
#[inline]
pub fn count<T>(counter: u128, result: T) -> T {
    let _ = increment_counter(counter);
    result
}

/// Increment the specified counter and then write the coverage report. This function normally wraps
/// the final expression in a `main()` function. There can be more than one statement, for example
/// if the `main()` has one or more `return` statements. In this case, all returns and the last
/// statement of `main()` (unless not reachable) should use this function.
#[stable(feature = "coverage", since = "1.44.0")]
pub fn count_and_report<T>(counter: u128, result: T) -> T {
    println!("Print the coverage counters:");
    let mut counters = increment_counter(counter).unwrap();
    for (counter, count) in counters.drain() {
        println!("Counter '{}' has count: {}", counter, count);
    }
    result
}
