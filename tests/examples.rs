#[macro_use]
extern crate duct;

#[test]
fn compile_test() {
    let mut error = false;
    for file in std::fs::read_dir("clippy_tests/examples").unwrap() {
        let file = file.unwrap().path();
        // only test *.rs files
        if file.extension().map_or(true, |file| file != "rs") {
            continue;
        }
        cmd!("touch", &file).run().unwrap();
        let output = file.with_extension("stderr");
        cmd!("cargo", "rustc", "-q", "--example", file.file_stem().unwrap(), "--", "-Dwarnings")
            .unchecked()
            .stderr(&output)
            .env("CLIPPY_DISABLE_WIKI_LINKS", "true")
            .dir("clippy_tests")
            .run()
            .unwrap();
        print!("testing {}... ", file.file_stem().unwrap().to_str().unwrap());
        if cmd!("git", "diff", "--exit-code", output).run().is_err() {
            error = true;
            println!("ERROR");
        } else {
            println!("ok");
        }
    }
    assert!(!error, "A test failed");
}
