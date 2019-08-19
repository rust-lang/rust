use walkdir::WalkDir;

use ra_tools::{gen_tests, generate_boilerplate, project_root, run_rustfmt, Verify};

#[test]
fn generated_grammar_is_fresh() {
    if let Err(error) = generate_boilerplate(Verify) {
        panic!("{}. Please update it by running `cargo gen-syntax`", error);
    }
}

#[test]
fn generated_tests_are_fresh() {
    if let Err(error) = gen_tests(Verify) {
        panic!("{}. Please update tests by running `cargo gen-tests`", error);
    }
}

#[test]
fn check_code_formatting() {
    if let Err(error) = run_rustfmt(Verify) {
        panic!("{}. Please format the code by running `cargo format`", error);
    }
}

#[test]
fn no_todo() {
    WalkDir::new(project_root().join("crates")).into_iter().for_each(|e| {
        let e = e.unwrap();
        if e.path().extension().map(|it| it != "rs").unwrap_or(true) {
            return;
        }
        if e.path().ends_with("tests/cli.rs") {
            return;
        }
        let text = std::fs::read_to_string(e.path()).unwrap();
        if text.contains("TODO") {
            panic!(
                "\nTODO markers should not be commited to the master branch,\n\
                 use FIXME instead\n\
                 {}\n",
                e.path().display(),
            )
        }
    })
}
