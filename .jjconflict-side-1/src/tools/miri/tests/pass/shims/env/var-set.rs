// Test a value set on the host (MIRI_ENV_VAR_TEST) and one that is not.
//@compile-flags: -Zmiri-env-set=MIRI_ENV_VAR_TEST=test_value_1 -Zmiri-env-set=TEST_VAR_2=test_value_2

fn main() {
    assert_eq!(std::env::var("MIRI_ENV_VAR_TEST"), Ok("test_value_1".to_owned()));
    assert_eq!(std::env::var("TEST_VAR_2"), Ok("test_value_2".to_owned()));
}
