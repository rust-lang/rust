// Test that rustdoc will properly load in a theme file and display it in the theme selector.

use run_make_support::{htmldocck, rustdoc, source_root};
use std::path::Path;

fn main() {
    let out_dir = Path::new("rustdoc-themes");
    let test_css = "test.css";

    let no_script =
        std::fs::read_to_string(source_root().join("src/librustdoc/html/static/css/noscript.css"))
            .unwrap();

    let mut test_content = String::new();
    let mut found_begin_light = false;
    for line in no_script.split('\n') {
        if line == "/* Begin theme: light */" {
            found_begin_light = true;
        } else if line == "/* End theme: light */" {
            break;
        } else if found_begin_light {
            test_content.push_str(line);
            test_content.push('\n');
        }
    }
    assert!(!test_content.is_empty());
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::write(&test_css, test_content).unwrap();

    rustdoc().output(&out_dir).input("foo.rs").arg("--theme").arg(&test_css).run();
    assert!(htmldocck().arg(out_dir).arg("foo.rs").status().unwrap().success());
}
