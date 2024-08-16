use run_make_support::path_helpers::read_dir_entries_recursive;
use run_make_support::rfs::read_to_string;
use run_make_support::{jzon, rustdoc};

fn main() {
    let out_dir = "out";
    rustdoc()
        .input("foo.rs")
        .arg("-Zunstable-options")
        .arg("--generate-redirect-map")
        .out_dir(&out_dir)
        .run();

    let mut found_file = false;
    read_dir_entries_recursive(&out_dir, |path| {
        if !found_file
            && path.is_file()
            && path.file_name().map(|name| name == "redirect-map.json").unwrap_or(false)
        {
            found_file = true;
            let generated = read_to_string(path);
            let expected = read_to_string("expected.json");
            let generated = jzon::parse(&generated).expect("failed to parse JSON");
            let expected = jzon::parse(&expected).expect("failed to parse JSON");

            let mut differences = Vec::new();
            for (key, expected_value) in expected.entries() {
                match generated.get(key) {
                    Some(value) => {
                        if expected_value != value {
                            differences.push(format!("values for key `{key}` don't match"));
                        }
                    }
                    None => differences.push(format!("missing key `{key}`")),
                }
            }
            for (key, data) in generated.entries() {
                if !expected.has_key(key) {
                    differences
                        .push(format!("Extra data not expected: key: `{key}`, data: `{data}`"));
                }
            }

            if !differences.is_empty() {
                eprintln!("Found differences in JSON files:");
                for diff in differences {
                    eprintln!("=> {diff}");
                }
                std::process::exit(1);
            }
        }
    });

    if !found_file {
        panic!("`redirect-map.json` file was not found");
    }
}
