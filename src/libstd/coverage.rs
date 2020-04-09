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
//! fn main() {
//!     let value = if true {
//!         std::coverage::count(1, {
//!             // any expression
//!             1000
//!         })
//!     } else {
//!         std::coverage::count(2, 500)
//!     };
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

// FIXME(richkadel): Thinking about ways of optimizing executing coverage-instrumented code, I may
// be able to replace the hashmap lookup with a simple vector index, for most invocations, by
// assigning an index to each hash, at execution time, and storing that index in an injected static
// at the site of each counter. If the static is initialized to an unused index (say, u64::MAX), I
// can pass it into the std::coverage::count() function as mutable, and update it to the next index
// available. The vector would contain the hashed region IDs and corresponding counts.
// I just need to be sure the change can be done atomically, and if the same counter is called from
// multiple threads before the index is assigned, that the behavior is reasonable. (I suppose it
// *could* even tolerate the possibility that two indexes were assigned. Printing the report could
// sum the counts for any identical coverage region IDs, if we need to allow for that.)
#[stable(feature = "coverage", since = "1.44.0")]
#[inline(always)]
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
        None => {
            counters.insert(counter, 1);
        }
    }
    lock
}

/// The statement may be the last statement of a block, the value of a `return` or `break`,
/// a match pattern arm statement that might not be in a block (but if it is, don't wrap both the
/// block and the last statement of the block), or a closure statement without braces.
///
/// ```no_run
///     # fn some_statement_with_or_without_semicolon() {}
///     std::coverage::count(234234, {some_statement_with_or_without_semicolon()})
/// ```
// Adding inline("always") for now, so I don't forget that some llvm experts thought
// it might not work to inject the llvm intrinsic inside this function if it is not
// inlined. Roughly speaking, LLVM lowering may change how the intrinsic and
// neighboring code is translated, in a way that behaves counterintuitively.
// (Note that "always" is still only a "recommendation", so this still might not
// address that problem.) Further investigation is needed, and an alternative
// solution would be to just expand code into the projected inline version at
// code injection time, rather than call this function at all.
#[stable(feature = "coverage", since = "1.44.0")]
#[inline(always)]
pub fn count<T>(counter: u128, result: T) -> T {
    // FIXME(richkadel): replace increment_counter with a call to the LLVM intrinsic:
    // ```
    // declare void @llvm.instrprof.increment(i8* <name>, i64 <hash>,
    //    i32 <num-counters>, i32 <index>)
    // ```
    // See: http://llvm.org/docs/LangRef.html#llvm-instrprof-increment-intrinsic
    let _ = increment_counter(counter);
    result
}

/// Increment the specified counter and then write the coverage report. This function normally wraps
/// the final expression in a `main()` function. There can be more than one statement, for example
/// if the `main()` has one or more `return` statements. In this case, all returns and the last
/// statement of `main()` (unless not reachable) should use this function.
#[stable(feature = "coverage", since = "1.44.0")]
#[inline(always)]
pub fn count_and_report<T>(counter: u128, result: T) -> T {
    println!("Print the coverage counters:");
    let mut counters = increment_counter(counter).unwrap();
    for (counter, count) in counters.drain() {
        println!("Counter '{}' has count: {}", counter, count);
    }
    result
}
