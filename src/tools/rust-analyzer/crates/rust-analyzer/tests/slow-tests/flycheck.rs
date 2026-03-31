use test_utils::skip_slow_tests;

use crate::support::Project;

#[test]
fn test_flycheck_diagnostics_for_unused_variable() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/main.rs
fn main() {
    let x = 1;
}
"#,
    )
    .with_config(serde_json::json!({
        "checkOnSave": true,
    }))
    .server()
    .wait_until_workspace_is_loaded();

    let diagnostics = server.wait_for_diagnostics();
    assert!(
        diagnostics.diagnostics.iter().any(|d| d.message.contains("unused variable")),
        "expected unused variable diagnostic, got: {:?}",
        diagnostics.diagnostics,
    );
}

#[test]
fn test_flycheck_diagnostic_cleared_after_fix() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/main.rs
fn main() {
    let x = 1;
}
"#,
    )
    .with_config(serde_json::json!({
        "checkOnSave": true,
    }))
    .server()
    .wait_until_workspace_is_loaded();

    // Wait for the unused variable diagnostic to appear.
    let diagnostics = server.wait_for_diagnostics();
    assert!(
        diagnostics.diagnostics.iter().any(|d| d.message.contains("unused variable")),
        "expected unused variable diagnostic, got: {:?}",
        diagnostics.diagnostics,
    );

    // Fix the code by removing the unused variable.
    server.write_file_and_save("src/main.rs", "fn main() {}\n".to_owned());

    // Wait for diagnostics to be cleared.
    server.wait_for_diagnostics_cleared();
}

#[test]
fn test_flycheck_diagnostic_with_override_command() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/main.rs
fn main() {}
"#,
    )
    .with_config(serde_json::json!({
        "checkOnSave": true,
        "check": {
            "overrideCommand": ["rustc", "--error-format=json", "$saved_file"]
        }
    }))
    .server()
    .wait_until_workspace_is_loaded();

    server.write_file_and_save("src/main.rs", "fn main() {\n    let x = 1;\n}\n".to_owned());

    let diagnostics = server.wait_for_diagnostics();
    assert!(
        diagnostics.diagnostics.iter().any(|d| d.message.contains("unused variable")),
        "expected unused variable diagnostic, got: {:?}",
        diagnostics.diagnostics,
    );
}
