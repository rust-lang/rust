//@compile-flags: -Zmiri-env-forward=MIRI_ENV_VAR_TEST

fn main() {
    assert_eq!(std::env::var("MIRI_ENV_VAR_TEST"), Ok("0".to_owned()));
}
