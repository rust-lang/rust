// Test the `suspicious_cargo_cfg_target_family_comparisons` lint.

//@ check-pass
//@ exec-env:CARGO_CFG_TARGET_FAMILY=unix

use std::env;

fn main() {
    // Check that direct comparisons warn.
    let is_unix = env::var("CARGO_CFG_TARGET_FAMILY").unwrap() == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // But that later usage doesn't warn.
    if is_unix {}

    // Assigning to local variable is fine.
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();

    // Using local in an `==` comparison.
    if target_family == "unix" {
        //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    }

    // Using local in a match.
    match &*target_family {
        //~^ WARN matching on `CARGO_CFG_TARGET_FAMILY` directly may break in the future
        "unix" => {}
        _ => {}
    }

    // Correct handling doesn't warn.
    if target_family.contains("unix") {}
    if target_family.split(',').any(|x| x == "unix") {}

    // Test supression.
    #[allow(suspicious_cargo_cfg_target_family_comparisons)]
    let _ = env::var("CARGO_CFG_TARGET_FAMILY").unwrap() == "unix";

    // Negative comparison.
    let _ = target_family != "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // Local variable propagation.
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    let target_family: &str = target_family.as_ref();
    let _ = target_family == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // Custom wrapper.
    fn get_and_track_env_var(env_var_name: &str) -> String {
        // This is actually unnecessary, Cargo already tracks changes to the target family, but it's
        // nonetheless a fairly common pattern.
        println!("cargo:rerun-if-env-changed={env_var_name}");
        env::var(env_var_name).unwrap()
    }
    let _ = get_and_track_env_var("CARGO_CFG_TARGET_FAMILY") == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // Various.
    let _ = ::std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap() == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var("CARGO_CFG_TARGET_FAMILY").expect("should be set") == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default() == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var_os("CARGO_CFG_TARGET_FAMILY").unwrap() == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var("CARGO_CFG_TARGET_FAMILY") == Ok("unix".to_string());
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var_os("CARGO_CFG_TARGET_FAMILY") == Some("unix".into());
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix");
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var_os("CARGO_CFG_TARGET_FAMILY").as_deref() == Some("unix".as_ref());
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
    let _ = env::var("CARGO_CFG_TARGET_FAMILY").ok().as_deref() == Some("unix".as_ref());
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    false_negatives();
    false_positives();
}

// This lint has many false negatives, the problem is intractable in the general case.
fn false_negatives() {
    // Cannot detect if the env var is not specified inline (such as when dynamically generated).
    let var = "CARGO_CFG_TARGET_FAMILY";
    let _ = env::var(var).unwrap() == "unix";

    // Cannot detect if env var value comes from somewhere more complex.
    fn get_env_var() -> String {
        env::var("CARGO_CFG_TARGET_FAMILY").unwrap()
    }
    let _ = get_env_var() == "unix";

    // Doesn't detect more complex expressions.
    let _ = std::convert::identity(env::var_os("CARGO_CFG_TARGET_FAMILY").unwrap()) == "unix";
    let _ = *Box::new(env::var_os("CARGO_CFG_TARGET_FAMILY").unwrap()) == "unix";

    // Doesn't detect variables that are initialized later.
    let later_init;
    later_init = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    if later_init == "unix" {}

    // Doesn't detect if placed inside a struct.
    struct Target {
        family: String,
    }
    let target = Target { family: env::var("CARGO_CFG_TARGET_FAMILY").unwrap() };
    if target.family == "unix" {}
}

// This lint also has false positives, these are probably unlikely to be hit in practice.
fn false_positives() {
    // Cannot detect later changes to assigned variable.
    let mut overwritten = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    if true {
        overwritten = "unix".to_string();
    }
    if overwritten == "unix" {}
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // Non-std::env::var usage.
    let _ = std::convert::identity("CARGO_CFG_TARGET_FAMILY") == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // Call unusual `Option`/`Result` method, and then compare that result.
    let _ = env::var_os("CARGO_CFG_TARGET_FAMILY").is_some() == true;
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    // Match with match arms that contains checks.
    match env::var("CARGO_CFG_TARGET_FAMILY") {
        //~^ WARN matching on `CARGO_CFG_TARGET_FAMILY` directly may break in the future
        Ok(os) if os.contains("unix") => {}
        _ => {}
    }

    // Unusual method call.
    trait Foo {
        fn returns_string(&self) -> &str {
            "unix"
        }
    }
    impl Foo for String {}
    let _ = env::var("CARGO_CFG_TARGET_FAMILY").unwrap().returns_string() == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
}
