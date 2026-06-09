use run_make_support::regex::Regex;
use run_make_support::{cwd, rfs, rustc};

fn main() {
    let invalid_characters = [".foo.rs", ".foo.bar", "+foo+bar.rs"];
    let re = Regex::new(r"invalid character.*in crate name:").unwrap();
    for f in invalid_characters {
        rfs::copy("foo.rs", f);
        let stderr = rustc().input(f).run_fail().stderr_utf8();
        assert!(re.is_match(&stderr));
    }

    rfs::copy("foo.rs", "-foo.rs");
    rustc()
        .input(cwd().join("-foo.rs"))
        .run_fail()
        .assert_stderr_contains("crate names cannot start with a `-`");
}
