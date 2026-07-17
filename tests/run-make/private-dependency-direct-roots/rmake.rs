use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().input("leaf.rs").run();
    rustc().input("left.rs").extern_("leaf", rust_lib_name("leaf")).run();
    rustc().input("right.rs").extern_("leaf", rust_lib_name("leaf")).run();

    let output = rustc()
        .input("root.rs")
        .arg("-Zunstable-options")
        .arg("--extern")
        .arg(format!("priv:right={}", rust_lib_name("right")))
        .arg("--extern")
        .arg(format!("priv:left={}", rust_lib_name("left")))
        .run_fail();

    output
        .assert_stderr_contains("type `Leaf` from private dependency 'leaf' in public interface")
        .assert_stderr_contains(
            "`leaf` is an indirect dependency reached through the private direct dependency `left`",
        )
        .assert_stderr_contains("consider marking `left` public if exposing this item is intended");
    assert!(!output.stderr_utf8().contains("private direct dependency `right`"));
}
