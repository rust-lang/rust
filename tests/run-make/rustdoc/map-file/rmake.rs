// This test ensures that all items from `foo` are correctly generated into the `redirect-map.json`
// file with `--generate-redirect-map` rustdoc option.

//@ needs-target-std

use run_make_support::rfs::read_to_string;
use run_make_support::{path, rustdoc, serde_json};

fn main() {
    let out_dir = "out";
    let crate_name = "foo";
    rustdoc()
        .input("foo.rs")
        .crate_name(crate_name)
        .arg("-Zunstable-options")
        .arg("--generate-redirect-map")
        .out_dir(&out_dir)
        .run();

    let generated = read_to_string(path(out_dir).join(crate_name).join("redirect-map.json"));
    let expected = read_to_string("expected.json");
    let generated: serde_json::Value =
        serde_json::from_str(&generated).expect("failed to parse JSON");
    let expected: serde_json::Value =
        serde_json::from_str(&expected).expect("failed to parse JSON");
    let expected = expected.as_object().unwrap();

    let mut differences = Vec::new();
    for (key, expected_value) in expected.iter() {
        match generated.get(key) {
            Some(value) => {
                if expected_value != value {
                    differences.push(format!(
                        "values for key `{key}` don't match: `{expected_value:?}` != `{value:?}`"
                    ));
                }
            }
            None => differences.push(format!("missing key `{key}`")),
        }
    }
    for (key, data) in generated.as_object().unwrap().iter() {
        if !expected.contains_key(key) {
            differences.push(format!("Extra data not expected: key: `{key}`, data: `{data}`"));
        }
    }

    if !differences.is_empty() {
        eprintln!("Found differences in JSON files:");
        for diff in differences {
            eprintln!("=> {diff}");
        }
        panic!("Found differences in JSON files");
    }
}
