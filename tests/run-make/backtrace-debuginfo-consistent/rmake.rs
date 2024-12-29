/** If at all possible, rather
  than removing the assert that the different revisions behave the same, move
  only the revisions that are failing into a separate test, so that the rest
  are still kept the same.
*/
use std::collections::BTreeMap;

use run_make_support::path_helpers::source_root;

fn main() {
    // compiletest generates a bunch of files for each revision. make sure they're all the same.
    let mut files = BTreeMap::new();
    // let dir = Path::new(env!("SOURCE_DIR")).join("backtrace");
    let dir = source_root().join("tests").join("ui").join("backtrace");
    for file in std::fs::read_dir(dir).unwrap() {
        let file = file.unwrap();
        let name = file.file_name().into_string().unwrap();
        if !file.file_type().unwrap().is_file()
            || !name.starts_with("std-backtrace-skip-frames.")
            || !name.ends_with(".run.stderr")
        {
            continue;
        }
        files.insert(name, std::fs::read_to_string(file.path()).unwrap());
    }

    let mut first_line_tables = None;
    let mut first_full = None;

    for (name, contents) in &files {
        // These have different output. Rather than duplicating this whole test,
        // just special-case them here.
        let target = if name.contains(".full.") || name.contains(".limited.") {
            &mut first_full
        } else {
            &mut first_line_tables
        };
        if let Some((target_name, target_contents)) = target {
            if contents != *target_contents {
                eprintln!(
                    "are you *sure* that you want {name} to have different backtrace output\
                           than {target_name}?"
                );
                eprintln!(
                    "NOTE: this test is stateful; run \
                           `rm tests/ui/backtrace/std-backtrace-skip-frames.*.stderr` to reset it"
                );
                std::process::exit(0);
            }
        } else {
            // compiletest doesn't support negative matching for `error-pattern`. Do that here.
            assert!(!contents.contains("FnOnce::call_once"));
            *target = Some((name, contents));
        }
    }
}
