/// ```
/// std::process::exit(1);
/// ```
fn bad_exit_code() {}

/// ```should_panic
/// std::process::exit(1);
/// ```
fn did_not_panic() {}

/// ```should_panic
/// panic!("yeay");
/// ```
fn did_panic() {}
