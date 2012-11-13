/*! Test definitions. The actual test runner is in std */

// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers seperated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.
pub type TestName = ~str;

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function fails then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into tasks.
pub type TestFn = fn~();

// The definition of a single test. A test runner will run a list of
// these.
pub type TestDesc = {
    name: TestName,
    testfn: TestFn,
    ignore: bool,
    should_fail: bool
};

// This exists to satisfy linkage. rustc's test pass generates a call to it,
// but we'll never use it. Instead driver.rs will call std::test::test_main
// directly.
pub fn test_main(_args: &[~str], _tests: &[TestDesc]) {
    fail
}