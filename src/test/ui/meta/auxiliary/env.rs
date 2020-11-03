// Check that aux builds can also use rustc-env, but environment is configured
// separately from the main test case.
//
// rustc-env:COMPILETEST_BAR=bar

pub fn test() {
    assert_eq!(option_env!("COMPILETEST_FOO"), None);
    assert_eq!(env!("COMPILETEST_BAR"), "bar");
}
