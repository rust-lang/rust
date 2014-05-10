// I have no idea what I'm doing with these attributes. Are we using
// semantic versioning? Some packages include their full github URL.
// Documentation for this stuff is extremely scarce.
#![crate_id = "quickcheck#0.1.0"]
#![crate_type = "lib"]
#![license = "UNLICENSE"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/quickcheck")]

//! This crate is a port of
//! [Haskell's QuickCheck](http://hackage.haskell.org/package/QuickCheck).
//!
//! For detailed examples, please see the
//! [README](https://github.com/BurntSushi/quickcheck).

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
    /// fn prop_reverse_reverse() {
    ///     fn revrev(xs: Vec<uint>) -> bool {
    ///         let rev = xs.clone().move_iter().rev().collect();
    ///         let revrev = rev.move_iter().rev().collect();
    ///         xs == revrev
    ///     }
    ///     check(revrev);
    /// }
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
    /// or fewer (both `fn` and `||` types) satisfy `Testable`. If you have
    /// functions to test with more than 3 parameters, please
    /// [file a bug](https://github.com/BurntSushi/quickcheck/issues) and
    /// I'll hopefully add it. (As of now, it would be very difficult to
    /// add your own implementation outside of `quickcheck`, since the
    /// functions that do shrinking are not public.)
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
