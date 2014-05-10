// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate is a port of [Haskell's
//! QuickCheck](http://hackage.haskell.org/package/QuickCheck) for
//! intelligent random testing.
//!
//! QuickCheck is a way to do property based testing using randomly generated
//! input. This crate comes with the ability to randomly generate and shrink
//! integers, floats, tuples, booleans, lists, strings, options and results.
//! All QuickCheck needs is a property function---it will then randomly generate
//! inputs to that function and call the property for each set of inputs. If the
//! property fails (whether by a runtime error like index out-of-bounds or by not
//! satisfying your property), the inputs are "shrunk" to find a smaller
//! counter-example.
//!
//! The shrinking strategies for lists and numbers use a binary search to cover
//! the input space quickly. (It should be the same strategy used in
//! [Koen Claessen's QuickCheck for
//! Haskell](http://hackage.haskell.org/package/QuickCheck).)
//!
//!
//! # Simple example
//!
//! Here's a complete working program that tests a function that
//! reverses a vector:
//!
//! ```rust
//! extern crate quickcheck;
//!
//! use quickcheck::quickcheck;
//!
//! fn reverse<T: Clone>(xs: &[T]) -> Vec<T> {
//!     let mut rev = vec!();
//!     for x in xs.iter() {
//!         rev.unshift(x.clone())
//!     }
//!     rev
//! }
//!
//! fn main() {
//!     fn prop(xs: Vec<int>) -> bool {
//!         xs == reverse(reverse(xs.as_slice()).as_slice())
//!     }
//!     quickcheck(prop);
//! }
//! ```
//!
//! # Discarding test results (or, properties are polymorphic!)
//!
//! Sometimes you want to test a property that only holds for a *subset* of the
//! possible inputs, so that when your property is given an input that is outside
//! of that subset, you'd discard it. In particular, the property should *neither*
//! pass nor fail on inputs outside of the subset you want to test. But properties
//! return boolean values---which either indicate pass or fail.
//!
//! To fix this, we need to take a step back and look at the type of the
//! `quickcheck` function:
//!
//! ```rust
//! # use quickcheck::Testable;
//! # fn main() {}
//! pub fn quickcheck<A: Testable>(f: A) {
//!     // elided
//! }
//! ```
//!
//! So `quickcheck` can test any value with a type that satisfies the `Testable`
//! trait. Great, so what is this `Testable` business?
//!
//! ```rust
//! # use quickcheck::{Gen, TestResult};
//!
//! pub trait Testable {
//!     fn result<G: Gen>(&self, &mut G) -> TestResult;
//! }
//! # pub fn main() {}
//! ```
//!
//! This trait states that a type is testable if it can produce a `TestResult`
//! given a source of randomness. (A `TestResult` stores information about the
//! results of a test, like whether it passed, failed or has been discarded.)
//!
//! Sure enough, `bool` satisfies the `Testable` trait:
//!
//! ```rust,ignore
//! impl Testable for bool {
//!     fn result<G: Gen>(&self, _: &mut G) -> TestResult {
//!         TestResult::from_bool(*self)
//!     }
//! }
//! ```
//!
//! But in the example, we gave a *function* to `quickcheck`. Yes, functions can
//! satisfy `Testable` too!
//!
//! ```rust,ignore
//! impl<A: Arbitrary + Show, B: Testable> Testable for fn(A) -> B {
//!     fn result<G: Gen>(&self, g: &mut G) -> TestResult {
//!         // elided
//!
//!     }
//! }
//! ```
//!
//! Which says that a function satisfies `Testable` if and only if it has a single
//! parameter type (whose values can be randomly generated and shrunk) and returns
//! any type (that also satisfies `Testable`). So a function with type
//! `fn(uint) -> bool` satisfies `Testable` since `uint` satisfies `Arbitrary` and
//! `bool` satisfies `Testable`.
//!
//! So to discard a test, we need to return something other than `bool`. What if we
//! just returned a `TestResult` directly? That should work, but we'll need to
//! make sure `TestResult` satisfies `Testable`:
//!
//! ```rust,ignore
//! impl Testable for TestResult {
//!     fn result<G: Gen>(&self, _: &mut G) -> TestResult { self.clone() }
//! }
//! ```
//!
//! Now we can test functions that return a `TestResult` directly.
//!
//! As an example, let's test our reverse function to make sure that the reverse of
//! a vector of length 1 is equal to the vector itself.
//!
//! ```rust
//! use quickcheck::{quickcheck, TestResult};
//!
//! # fn reverse<T: Clone>(xs: &[T]) -> Vec<T> {
//! #     let mut rev = vec!();
//! #     for x in xs.iter() { rev.unshift(x.clone()) }
//! #     rev
//! # }
//! fn prop(xs: Vec<int>) -> TestResult {
//!     if xs.len() != 1 {
//!         return TestResult::discard()
//!     }
//!     TestResult::from_bool(xs == reverse(xs.as_slice()))
//! }
//! quickcheck(prop);
//! ```
//!
//! So now our property returns a `TestResult`, which allows us to
//! encode a bit more information. There are a few more [convenience
//! functions defined for the `TestResult`
//! type](struct.TestResult.html).  For example, we can't just return
//! a `bool`, so we convert a `bool` value to a `TestResult`.
//!
//! (The ability to discard tests allows you to get similar functionality as
//! Haskell's `==>` combinator.)
//!
//! N.B. Since discarding a test means it neither passes nor fails,
//! `quickcheck` will try to replace the discarded test with a fresh
//! one. However, if your condition is seldom met, it's possible that
//! `quickcheck` will have to settle for running fewer tests than
//! usual. By default, if `quickcheck` can't find `100` valid tests
//! after trying `10,000` times, then it will give up.  This parameter
//! may be changed using
//! [`quickcheck_config`](fn.quickcheck_config.html).
//!
//!
//! # Shrinking
//!
//! Shrinking is a crucial part of QuickCheck that simplifies counter-examples for
//! your properties automatically. For example, if you erroneously defined a
//! function for reversing vectors as:
//!
//! ```rust,should_fail
//! use std::iter;
//! use quickcheck::quickcheck;
//!
//! fn reverse<T: Clone>(xs: &[T]) -> Vec<T> {
//!     let mut rev = vec!();
//!     for i in iter::range(1, xs.len()) {
//!         rev.unshift(xs[i].clone())
//!     }
//!     rev
//! }
//!
//! /// A property that tests that reversing twice is the same as the
//! /// original vector
//! fn prop(xs: Vec<int>) -> bool {
//!     xs == reverse(reverse(xs.as_slice()).as_slice())
//! }
//! quickcheck(prop);
//! ```
//!
//! Then without shrinking, you might get a counter-example like:
//!
//! ```notrust
//! [quickcheck] TEST FAILED. Arguments: ([-17, 13, -12, 17, -8, -10, 15, -19,
//! -19, -9, 11, -5, 1, 19, -16, 6])
//! ```
//!
//! Which is pretty mysterious. But with shrinking enabled, you're nearly
//! guaranteed to get this counter-example every time:
//!
//! ```notrust
//! [quickcheck] TEST FAILED. Arguments: ([0])
//! ```
//!
//! Which is going to be much easier to debug.
//!
//!
//! # Case study: The Sieve of Eratosthenes
//!
//! The [Sieve of Eratosthenes](http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)
//! is a simple and elegant way to find all primes less than or equal to `N`.
//! Briefly, the algorithm works by allocating an array with `N` slots containing
//! booleans. Slots marked with `false` correspond to prime numbers (or numbers
//! not known to be prime while building the sieve) and slots marked with `true`
//! are known to not be prime. For each `n`, all of its multiples in this array
//! are marked as true. When all `n` have been checked, the numbers marked `false`
//! are returned as the primes.
//!
//! As you might imagine, there's a lot of potential for off-by-one
//! errors, which makes it ideal for randomized testing. So let's take
//! a look at an implementation and see if we can spot the bug:
//!
//! ```rust,should_fail
//! use std::iter;
//! use quickcheck::quickcheck;
//!
//! fn sieve(n: uint) -> Vec<uint> {
//!     if n <= 1 {
//!         return vec!()
//!     }
//!
//!     let mut marked = Vec::from_fn(n+1, |_| false);
//!     *marked.get_mut(0) = true;
//!     *marked.get_mut(1) = true;
//!     *marked.get_mut(2) = false;
//!     for p in iter::range(2, n) {
//!         for i in iter::range_step(2 * p, n, p) { // whoops!
//!             *marked.get_mut(i) = true;
//!         }
//!     }
//!     let mut primes = vec!();
//!     for (i, m) in marked.iter().enumerate() {
//!         if !m { primes.push(i) }
//!     }
//!     primes
//! }
//!
//! /*
//! Let's try it on a few inputs by hand:
//!
//! sieve(3) => [2, 3]
//! sieve(5) => [2, 3, 5]
//! sieve(8) => [2, 3, 5, 7, 8] # !!!
//!
//! Something has gone wrong! But where? The bug is rather subtle, but it's an
//! easy one to make. It's OK if you can't spot it, because we're going to use
//! QuickCheck to help us track it down.
//!
//! Even before looking at some example outputs, it's good to try and come up with
//! some *properties* that are always satisfiable by the output of the function. An
//! obvious one for the prime number sieve is to check if all numbers returned are
//! prime. For that, we'll need an `is_prime` function:
//! */
//!
//! /// Check if `n` is prime by trial division.
//! fn is_prime(n: uint) -> bool {
//!     // some base cases:
//!     if n == 0 || n == 1 {
//!         return false
//!     } else if n == 2 {
//!         return true
//!     }
//!
//!     // run through the numbers in `[2, sqrt(n)]` (inclusive) to
//!     // see if any divide `n`.
//!     let max_possible = (n as f64).sqrt().ceil() as uint;
//!     for i in iter::range_inclusive(2, max_possible) {
//!         if n % i == 0 {
//!             return false
//!         }
//!     }
//!     return true
//! }
//!
//! /*
//! Now we can write our quickcheck property.
//! */
//!
//! fn prop_all_prime(n: uint) -> bool {
//!     let primes = sieve(n);
//!     primes.iter().all(|&i| is_prime(i))
//! }
//!
//! fn main() {
//!     // invoke quickcheck
//!     quickcheck(prop_all_prime);
//! }
//! ```
//!
//! The output of running this program has this message:
//!
//! ```notrust
//! [quickcheck] TEST FAILED. Arguments: (4)
//! ```
//!
//! Which says that `sieve` failed the `prop_all_prime` test when given `n = 4`.
//! Because of shrinking, it was able to find a (hopefully) minimal counter-example
//! for our property.
//!
//! With such a short counter-example, it's hopefully a bit easier to narrow down
//! where the bug is. Since `4` is returned, it's likely never marked as being not
//! prime. Since `4` is a multiple of `2`, its slot should be marked as `true` when
//! `p = 2` on these lines:
//!
//! ```rust
//! # use std::iter;
//! # let (p, n, mut marked) = (2u, 4, vec!(false, false, false, false));
//! for i in iter::range_step(2 * p, n, p) {
//!     *marked.get_mut(i) = true;
//! }
//! ```
//!
//! Ah! But does the `range_step` function include `n`? Its documentation says
//!
//! > Return an iterator over the range [start, stop) by step. It handles overflow
//! > by stopping.
//!
//! Shucks. The `range_step` function will never yield `4` when `n = 4`. We could
//! use `n + 1`, but the `std::iter` crate also has a
//! [`range_step_inclusive`](../std/iter/fn.range_step_inclusive.html)
//! which seems clearer.
//!
//! Changing the call to `range_step_inclusive` results in `sieve` passing all
//! tests for the `prop_all_prime` property.
//!
//! In addition, if our bug happened to result in an index out-of-bounds error,
//! then `quickcheck` can handle it just like any other failure---including
//! shrinking on failures caused by runtime errors.
//!
//!
//! # What's not in this port of QuickCheck?
//!
//! The key features have been captured, but there are still things
//! missing:
//!
//! * As of now, only functions with 3 or fewer parameters can be quickchecked.
//! This limitation can be lifted to some `N`, but requires an implementation
//! for each `n` of the `Testable` trait.
//! * Functions that fail because of a stack overflow are not caught
//!   by QuickCheck.  Therefore, such failures will not have a witness
//!   attached to them.
//! * `Coarbitrary` does not exist in any form in this package.
//!
//! # Laziness
//!
//! A key aspect for writing good shrinkers is a good lazy
//! abstraction. For this, iterators were chosen as the
//! representation. The insistence on this point has resulted in the
//! use of an existential type.
//!
//! Note though that the shrinkers for lists and integers are not lazy. Their
//! algorithms are more complex, so it will take a bit more work to get them to
//! use iterators like the rest of the shrinking strategies.

#![crate_id = "quickcheck#0.11-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![experimental]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![feature(phase)]

extern crate collections;
#[phase(syntax, link)] extern crate log;
extern crate rand;

pub use arbitrary::{Arbitrary, Gen, StdGen, Shrinker, gen, empty_shrinker, single_shrinker};
pub use tester::{Testable, TestResult, Config};
pub use tester::{quickcheck, quickcheck_config, quicktest, quicktest_config};
pub use tester::{DEFAULT_CONFIG, DEFAULT_SIZE};

mod arbitrary;

mod tester {
    use std::fmt::Show;
    use std::iter;
    use rand::task_rng;
    use super::{Arbitrary, Gen, Shrinker, gen};
    use tester::trap::safe;

    /// Default size hint used in `quickcheck` for sampling from a random
    /// distribution.
    pub static DEFAULT_SIZE: uint = 100;

    /// Default configuration used in `quickcheck`.
    pub static DEFAULT_CONFIG: Config = Config{
        tests: 100,
        max_tests: 10000,
    };

    /// Does randomized testing on `f` and produces a possibly minimal
    /// witness for test failures.
    ///
    /// This function is equivalent to calling `quickcheck_config` with
    /// `DEFAULT_CONFIG` and a `Gen` with size `DEFAULT_SIZE`.
    ///
    /// As of now, it is intended for `quickcheck` to be used inside Rust's
    /// unit testing system. For example, to check if
    /// `reverse(reverse(xs)) == xs`, you could use:
    ///
    /// ```rust
    /// use quickcheck::quickcheck;
    ///
    /// fn prop_reverse_reverse() {
    ///     fn revrev(xs: Vec<uint>) -> bool {
    ///         let mut revrev = xs.clone();
    ///         revrev.reverse();
    ///         revrev.reverse();
    ///
    ///         xs == revrev
    ///     }
    ///     quickcheck(revrev);
    /// }
    /// # fn main() { prop_reverse_reverse() }
    /// ```
    ///
    /// In particular, `quickcheck` will call `fail!` if it finds a
    /// test failure. The failure message will include a witness to the
    /// failure.
    pub fn quickcheck<A: Testable>(f: A) {
        let g = &mut gen(task_rng(), DEFAULT_SIZE);
        quickcheck_config(DEFAULT_CONFIG, g, f)
    }

    /// Does randomized testing on `f` with the given config and produces a
    /// possibly minimal witness for test failures.
    pub fn quickcheck_config<A: Testable, G: Gen>(c: Config, g: &mut G, f: A) {
        match quicktest_config(c, g, f) {
            Ok(ntests) => debug!("[quickcheck] Passed {:u} tests.", ntests),
            Err(r) => fail!(r.failed_msg()),
        }
    }

    /// Like `quickcheck`, but returns either the number of tests passed
    /// or a witness of failure.
    pub fn quicktest<A: Testable>(f: A) -> Result<uint, TestResult> {
        let g = &mut gen(task_rng(), DEFAULT_SIZE);
        quicktest_config(DEFAULT_CONFIG, g, f)
    }

    /// Like `quickcheck_config`, but returns either the number of tests passed
    /// or a witness of failure.
    pub fn quicktest_config<A: Testable, G: Gen>
                           (c: Config, g: &mut G, f: A)
                           -> Result<uint, TestResult> {
        let mut ntests: uint = 0;
        for _ in iter::range(0, c.max_tests) {
            if ntests >= c.tests {
                break
            }
            let r = f.result(g);
            match r.status {
                Pass => ntests = ntests + 1,
                Discard => continue,
                Fail => {
                    return Err(r)
                }
            }
        }
        Ok(ntests)
    }

    /// Config contains various parameters for controlling automated testing.
    ///
    /// Note that the distribution of random values is controlled by the
    /// generator passed to `quickcheck_config`.
    pub struct Config {
        /// The number of tests to run on a function where the result is
        /// either a pass or a failure. (i.e., This doesn't include discarded
        /// test results.)
        pub tests: uint,

        /// The maximum number of tests to run for each function including
        /// discarded test results.
        pub max_tests: uint,
    }

    /// Describes the status of a single instance of a test.
    ///
    /// All testable things must be capable of producing a `TestResult`.
    #[deriving(Clone, Show)]
    pub struct TestResult {
        status: Status,
        arguments: Vec<~str>,
        err: ~str,
    }

    /// Whether a test has passed, failed or been discarded.
    #[deriving(Clone, Show)]
    enum Status { Pass, Fail, Discard }

    impl TestResult {
        /// Produces a test result that indicates the current test has passed.
        pub fn passed() -> TestResult { TestResult::from_bool(true) }

        /// Produces a test result that indicates the current test has failed.
        pub fn failed() -> TestResult { TestResult::from_bool(false) }

        /// Produces a test result that indicates failure from a runtime
        /// error.
        pub fn error(msg: &str) -> TestResult {
            let mut r = TestResult::from_bool(false);
            r.err = msg.to_owned();
            r
        }

        /// Produces a test result that instructs `quickcheck` to ignore it.
        /// This is useful for restricting the domain of your properties.
        /// When a test is discarded, `quickcheck` will replace it with a
        /// fresh one (up to a certain limit).
        pub fn discard() -> TestResult {
            TestResult { status: Discard, arguments: vec!(), err: "".to_owned(), }
        }

        /// Converts a `bool` to a `TestResult`. A `true` value indicates that
        /// the test has passed and a `false` value indicates that the test
        /// has failed.
        pub fn from_bool(b: bool) -> TestResult {
            TestResult {
                status: if b { Pass } else { Fail },
                arguments: vec!(),
                err: "".to_owned(),
            }
        }

        /// Returns `true` if and only if this test result describes a failing
        /// test.
        pub fn is_failure(&self) -> bool {
            match self.status {
                Fail => true,
                Pass|Discard => false,
            }
        }

        /// Returns `true` if and only if this test result describes a failing
        /// test as a result of a run time error.
        pub fn is_error(&self) -> bool {
            return self.is_failure() && self.err.len() > 0
        }

        fn failed_msg(&self) -> ~str {
            if self.err.len() == 0 {
                return format!(
                    "[quickcheck] TEST FAILED. Arguments: ({})",
                    self.arguments.connect(", "))
            } else {
                return format!(
                    "[quickcheck] TEST FAILED (runtime error). \
                    Arguments: ({})\nError: {}",
                    self.arguments.connect(", "), self.err)
            }
        }
    }

    /// `Testable` describes types (e.g., a function) whose values can be
    /// tested.
    ///
    /// Anything that can be tested must be capable of producing a `TestResult`
    /// given a random number generator. This is trivial for types like `bool`,
    /// which are just converted to either a passing or failing test result.
    ///
    /// For functions, an implementation must generate random arguments
    /// and potentially shrink those arguments if they produce a failure.
    ///
    /// It's unlikely that you'll have to implement this trait yourself.
    /// This comes with a caveat: currently, only functions with 3 parameters
    /// or fewer (both `fn` and `||` types) satisfy `Testable`.
    pub trait Testable : Send {
        fn result<G: Gen>(&self, &mut G) -> TestResult;
    }

    impl Testable for bool {
        fn result<G: Gen>(&self, _: &mut G) -> TestResult {
            TestResult::from_bool(*self)
        }
    }

    impl Testable for TestResult {
        fn result<G: Gen>(&self, _: &mut G) -> TestResult { self.clone() }
    }

    impl<A: Testable> Testable for Result<A, ~str> {
        fn result<G: Gen>(&self, g: &mut G) -> TestResult {
            match *self {
                Ok(ref r) => r.result(g),
                Err(ref err) => TestResult::error(*err),
            }
        }
    }

    impl<T: Testable> Testable for fn() -> T {
        fn result<G: Gen>(&self, g: &mut G) -> TestResult {
            shrink(g, Zero::<(), (), (), T>(*self))
        }
    }

    impl<A: AShow, T: Testable> Testable for fn(A) -> T {
        fn result<G: Gen>(&self, g: &mut G) -> TestResult {
            shrink(g, One::<A, (), (), T>(*self))
        }
    }

    impl<A: AShow, B: AShow, T: Testable> Testable for fn(A, B) -> T {
        fn result<G: Gen>(&self, g: &mut G) -> TestResult {
            shrink(g, Two::<A, B, (), T>(*self))
        }
    }

    impl<A: AShow, B: AShow, C: AShow, T: Testable>
        Testable for fn(A, B, C) -> T {
        fn result<G: Gen>(&self, g: &mut G) -> TestResult {
            shrink(g, Three::<A, B, C, T>(*self))
        }
    }

    enum Fun<A, B, C, T> {
        Zero(fn() -> T),
        One(fn(A) -> T),
        Two(fn(A, B) -> T),
        Three(fn(A, B, C) -> T),
    }

    impl<A: AShow, B: AShow, C: AShow, T: Testable> Fun<A, B, C, T> {
        fn call<G: Gen>(self, g: &mut G,
                        a: Option<&A>, b: Option<&B>, c: Option<&C>)
                       -> TestResult {
            match self {
                Zero(f) => safe(proc() { f() }).result(g),
                One(f) => {
                    let a = a.unwrap();
                    let oa = box a.clone();
                    let mut r = safe(proc() { f(*oa) }).result(g);
                    if r.is_failure() {
                        r.arguments = vec!(a.to_str());
                    }
                    r
                },
                Two(f) => {
                    let (a, b) = (a.unwrap(), b.unwrap());
                    let (oa, ob) = (box a.clone(), box b.clone());
                    let mut r = safe(proc() { f(*oa, *ob) }).result(g);
                    if r.is_failure() {
                        r.arguments = vec!(a.to_str(), b.to_str());
                    }
                    r
                },
                Three(f) => {
                    let (a, b, c) = (a.unwrap(), b.unwrap(), c.unwrap());
                    let (oa, ob, oc) = (box a.clone(), box b.clone(), box c.clone());
                    let mut r = safe(proc() { f(*oa, *ob, *oc) }).result(g);
                    if r.is_failure() {
                        r.arguments = vec!(a.to_str(), b.to_str(), c.to_str());
                    }
                    r
                },
            }
        }
    }

    fn shrink<G: Gen, A: AShow, B: AShow, C: AShow, T: Testable>
             (g: &mut G, fun: Fun<A, B, C, T>)
             -> TestResult {
        let (a, b, c): (A, B, C) = arby(g);
        let r = fun.call(g, Some(&a), Some(&b), Some(&c));
        match r.status {
            Pass|Discard => r,
            Fail => shrink_failure(g, (a, b, c).shrink(), fun).unwrap_or(r),
        }
    }

    fn shrink_failure<G: Gen, A: AShow, B: AShow, C: AShow, T: Testable>
                     (g: &mut G, mut shrinker: Box<Shrinker<(A, B, C)>>,
                      fun: Fun<A, B, C, T>)
                     -> Option<TestResult> {
        for (a, b, c) in shrinker {
            let r = fun.call(g, Some(&a), Some(&b), Some(&c));
            match r.status {
                // The shrunk value does not witness a failure, so
                // throw it away.
                Pass|Discard => continue,

                // The shrunk value *does* witness a failure, so keep trying
                // to shrink it.
                Fail => {
                    let shrunk = shrink_failure(g, (a, b, c).shrink(), fun);

                    // If we couldn't witness a failure on any shrunk value,
                    // then return the failure we already have.
                    return Some(shrunk.unwrap_or(r))
                },
            }
        }
        None
    }

    #[cfg(quickfail)]
    mod trap {
        pub fn safe<T: Send>(fun: proc() -> T) -> Result<T, ~str> {
            Ok(fun())
        }
    }

    #[cfg(not(quickfail))]
    mod trap {
        use std::comm::channel;
        use std::io::{ChanReader, ChanWriter};
        use std::task::TaskBuilder;

        // This is my bright idea for capturing runtime errors caused by a
        // test. Actually, it looks like rustc uses a similar approach.
        // The problem is, this is used for *each* test case passed to a
        // property, whereas rustc does it once for each test.
        //
        // I'm not entirely sure there's much of an alternative either.
        // We could launch a single task and pass arguments over a channel,
        // but the task would need to be restarted if it failed due to a
        // runtime error. Since these are rare, it'd probably be more efficient
        // then this approach, but it would also be more complex.
        //
        // Moreover, this feature seems to prevent an implementation of
        // Testable for a stack closure type. *sigh*
        pub fn safe<T: Send>(fun: proc():Send -> T) -> Result<T, ~str> {
            let (send, recv) = channel();
            let stdout = ChanWriter::new(send.clone());
            let stderr = ChanWriter::new(send);
            let mut reader = ChanReader::new(recv);

            let mut t = TaskBuilder::new();
            t.opts.name = Some(("safefn".to_owned()).into_maybe_owned());
            t.opts.stdout = Some(box stdout as Box<Writer:Send>);
            t.opts.stderr = Some(box stderr as Box<Writer:Send>);

            match t.try(fun) {
                Ok(v) => Ok(v),
                Err(_) => {
                    let s = reader.read_to_str().unwrap();
                    Err(s.trim().into_owned())
                }
            }
        }
    }

    /// Convenient aliases.
    trait AShow : Arbitrary + Show {}
    impl<A: Arbitrary + Show> AShow for A {}
    fn arby<A: Arbitrary, G: Gen>(g: &mut G) -> A { Arbitrary::arbitrary(g) }
}

#[cfg(test)]
mod test {
    use std::cmp::TotalOrd;
    use std::iter;
    use rand::task_rng;
    use super::{Config, Testable, TestResult, gen};
    use super::{quickcheck_config, quicktest_config};

    static SIZE: uint = 100;
    static CONFIG: Config = Config {
        tests: 100,
        max_tests: 10000,
    };

    fn qcheck<A: Testable>(f: A) {
        quickcheck_config(CONFIG, &mut gen(task_rng(), SIZE), f)
    }

    fn qtest<A: Testable>(f: A) -> Result<uint, TestResult> {
        quicktest_config(CONFIG, &mut gen(task_rng(), SIZE), f)
    }

    #[test]
    fn prop_oob() {
        fn prop() -> bool {
            let zero: Vec<bool> = vec!();
            *zero.get(0)
        }
        match qtest(prop) {
            Ok(n) => fail!("prop_oob should fail with a runtime error \
                            but instead it passed {} tests.", n),
            _ => return,
        }
    }

    #[test]
    fn prop_reverse_reverse() {
        fn prop(xs: Vec<uint>) -> bool {
            let rev: Vec<uint> = xs.clone().move_iter().rev().collect();
            let revrev = rev.move_iter().rev().collect();
            xs == revrev
        }
        qcheck(prop);
    }

    #[test]
    fn reverse_single() {
        fn prop(xs: Vec<uint>) -> TestResult {
            if xs.len() != 1 {
                return TestResult::discard()
            }
            return TestResult::from_bool(
                xs == xs.clone().move_iter().rev().collect()
            )
        }
        qcheck(prop);
    }

    #[test]
    fn reverse_app() {
        fn prop(xs: Vec<uint>, ys: Vec<uint>) -> bool {
            let app = xs.clone().append(ys.as_slice());
            let app_rev: Vec<uint> = app.move_iter().rev().collect();

            let rxs = xs.move_iter().rev().collect();
            let mut rev_app = ys.move_iter().rev().collect::<Vec<uint>>();
            rev_app.push_all_move(rxs);

            app_rev == rev_app
        }
        qcheck(prop);
    }

    #[test]
    fn max() {
        fn prop(x: int, y: int) -> TestResult {
            if x > y {
                return TestResult::discard()
            } else {
                return TestResult::from_bool(::std::cmp::max(x, y) == y)
            }
        }
        qcheck(prop);
    }

    #[test]
    fn sort() {
        fn prop(mut xs: Vec<int>) -> bool {
            xs.sort_by(|x, y| x.cmp(y));
            let upto = if xs.len() == 0 { 0 } else { xs.len()-1 };
            for i in iter::range(0, upto) {
                if xs.get(i) > xs.get(i+1) {
                    return false
                }
            }
            true
        }
        qcheck(prop);
    }

    #[test]
    #[should_fail]
    fn sieve_of_eratosthenes() {
        fn sieve(n: uint) -> Vec<uint> {
            if n <= 1 {
                return vec!()
            }

            let mut marked = Vec::from_fn(n+1, |_| false);
            *marked.get_mut(0) = true;
            *marked.get_mut(1) = true;
            *marked.get_mut(2) = false;
            for p in iter::range(2, n) {
                for i in iter::range_step(2 * p, n, p) { // whoops!
                    *marked.get_mut(i) = true;
                }
            }
            let mut primes = vec!();
            for (i, m) in marked.iter().enumerate() {
                if !m { primes.push(i) }
            }
            primes
        }

        fn prop(n: uint) -> bool {
            let primes = sieve(n);
            primes.iter().all(|&i| is_prime(i))
        }
        fn is_prime(n: uint) -> bool {
            if n == 0 || n == 1 {
                return false
            } else if n == 2 {
                return true
            }

            let max_possible = (n as f64).sqrt().ceil() as uint;
            for i in iter::range_inclusive(2, max_possible) {
                if n % i == 0 {
                    return false
                }
            }
            return true
        }
        qcheck(prop);
    }
}
