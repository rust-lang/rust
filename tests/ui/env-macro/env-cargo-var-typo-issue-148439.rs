//@ edition: 2021

// Regression test for issue #148439
// Ensure that when using misspelled Cargo environment variables in env!(),

fn test_cargo_package_version() {
    let _ = env!("CARGO_PACKAGE_VERSION");
    //~^ ERROR environment variable `CARGO_PACKAGE_VERSION` not defined at compile time
    //~| HELP there is a similar Cargo environment variable: `CARGO_PKG_VERSION`
}

fn test_cargo_package_name() {
    let _ = env!("CARGO_PACKAGE_NAME");
    //~^ ERROR environment variable `CARGO_PACKAGE_NAME` not defined at compile time
    //~| HELP there is a similar Cargo environment variable: `CARGO_PKG_NAME`
}

fn test_cargo_package_authors() {
    let _ = env!("CARGO_PACKAGE_AUTHORS");
    //~^ ERROR environment variable `CARGO_PACKAGE_AUTHORS` not defined at compile time
    //~| HELP there is a similar Cargo environment variable: `CARGO_PKG_AUTHORS`
}

fn test_cargo_manifest_directory() {
    let _ = env!("CARGO_MANIFEST_DIRECTORY");
    //~^ ERROR environment variable `CARGO_MANIFEST_DIRECTORY` not defined at compile time
    //~| HELP there is a similar Cargo environment variable: `CARGO_MANIFEST_DIR`
}

fn test_cargo_pkg_version_typo() {
    let _ = env!("CARGO_PKG_VERSIO");
    //~^ ERROR environment variable `CARGO_PKG_VERSIO` not defined at compile time
    //~| HELP there is a similar Cargo environment variable: `CARGO_PKG_VERSION`
}

fn test_non_cargo_var() {
    // Non-Cargo variable should get different help message
    let _ = env!("MY_CUSTOM_VAR");
    //~^ ERROR environment variable `MY_CUSTOM_VAR` not defined at compile time
    //~| HELP use `std::env::var("MY_CUSTOM_VAR")` to read the variable at run time
}

fn test_cargo_unknown_var() {
    // Cargo-prefixed but not similar to any known variable
    let _ = env!("CARGO_SOMETHING_TOTALLY_UNKNOWN");
    //~^ ERROR environment variable `CARGO_SOMETHING_TOTALLY_UNKNOWN` not defined at compile time
    //~| HELP Cargo sets build script variables at run time. Use `std::env::var("CARGO_SOMETHING_TOTALLY_UNKNOWN")` instead
}

fn main() {}
