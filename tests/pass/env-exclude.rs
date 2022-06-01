// compile-flags: -Zmiri-disable-isolation -Zmiri-env-exclude=MIRI_ENV_VAR_TEST

fn main() {
    assert!(std::env::var("MIRI_ENV_VAR_TEST").is_err());
}
