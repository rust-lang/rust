use std::env;

fn main() {
    // Test that miri environment is isolated when communication is disabled.
    // (`MIRI_ENV_VAR_TEST` is set by the test harness.)
    assert_eq!(env::var("MIRI_ENV_VAR_TEST"), Err(env::VarError::NotPresent));

    // Test base state.
    println!("{:#?}", env::vars().collect::<Vec<_>>());
    assert_eq!(env::var("MIRI_TEST"), Err(env::VarError::NotPresent));

    // Set the variable.
    env::set_var("MIRI_TEST", "the answer");
    assert_eq!(env::var("MIRI_TEST"), Ok("the answer".to_owned()));
    println!("{:#?}", env::vars().collect::<Vec<_>>());

    // Change the variable.
    env::set_var("MIRI_TEST", "42");
    assert_eq!(env::var("MIRI_TEST"), Ok("42".to_owned()));
    println!("{:#?}", env::vars().collect::<Vec<_>>());

    // Remove the variable.
    env::remove_var("MIRI_TEST");
    assert_eq!(env::var("MIRI_TEST"), Err(env::VarError::NotPresent));
    println!("{:#?}", env::vars().collect::<Vec<_>>());
}
