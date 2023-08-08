use std::fs;
use std::path::Path;

pub struct MiroptTestFiles {
    pub expected_file: std::path::PathBuf,
    pub from_file: String,
    pub to_file: Option<String>,
    /// Vec of passes under test to be dumped
    pub passes: Vec<String>,
}

pub enum PanicStrategy {
    Unwind,
    Abort,
}

pub fn output_file_suffix(
    testfile: &Path,
    bit_width: u32,
    panic_strategy: PanicStrategy,
) -> String {
    let mut each_bit_width = false;
    let mut each_panic_strategy = false;
    for line in fs::read_to_string(testfile).unwrap().lines() {
        if line == "// EMIT_MIR_FOR_EACH_BIT_WIDTH" {
            each_bit_width = true;
        }
        if line == "// EMIT_MIR_FOR_EACH_PANIC_STRATEGY" {
            each_panic_strategy = true;
        }
    }

    let mut suffix = String::new();
    if each_bit_width {
        suffix.push_str(&format!(".{}bit", bit_width));
    }
    if each_panic_strategy {
        match panic_strategy {
            PanicStrategy::Unwind => suffix.push_str(".panic-unwind"),
            PanicStrategy::Abort => suffix.push_str(".panic-abort"),
        }
    }
    suffix
}

pub fn files_for_miropt_test(
    testfile: &std::path::Path,
    bit_width: u32,
    panic_strategy: PanicStrategy,
) -> Vec<MiroptTestFiles> {
    let mut out = Vec::new();
    let test_file_contents = fs::read_to_string(&testfile).unwrap();

    let test_dir = testfile.parent().unwrap();
    let test_crate = testfile.file_stem().unwrap().to_str().unwrap().replace('-', "_");

    let suffix = output_file_suffix(testfile, bit_width, panic_strategy);

    for l in test_file_contents.lines() {
        if l.starts_with("// EMIT_MIR ") {
            let test_name = l.trim_start_matches("// EMIT_MIR ").trim();
            let mut test_names = test_name.split(' ');
            // sometimes we specify two files so that we get a diff between the two files
            let test_name = test_names.next().unwrap();
            let mut expected_file;
            let from_file;
            let to_file;
            let mut passes = Vec::new();

            if test_name.ends_with(".diff") {
                let trimmed = test_name.trim_end_matches(".diff");
                passes.push(trimmed.split('.').last().unwrap().to_owned());
                let test_against = format!("{}.after.mir", trimmed);
                from_file = format!("{}.before.mir", trimmed);
                expected_file = format!("{}{}.diff", trimmed, suffix);
                assert!(test_names.next().is_none(), "two mir pass names specified for MIR diff");
                to_file = Some(test_against);
            } else if let Some(first_pass) = test_names.next() {
                let second_pass = test_names.next().unwrap();
                if let Some((first_pass_name, _)) = first_pass.split_once('.') {
                    passes.push(first_pass_name.to_owned());
                }
                if let Some((second_pass_name, _)) = second_pass.split_once('.') {
                    passes.push(second_pass_name.to_owned());
                }
                assert!(test_names.next().is_none(), "three mir pass names specified for MIR diff");

                expected_file =
                    format!("{}{}.{}-{}.diff", test_name, suffix, first_pass, second_pass);
                let second_file = format!("{}.{}.mir", test_name, second_pass);
                from_file = format!("{}.{}.mir", test_name, first_pass);
                to_file = Some(second_file);
            } else {
                let ext_re = regex::Regex::new(r#"(\.(mir|dot|html))$"#).unwrap();
                let cap = ext_re
                    .captures_iter(test_name)
                    .next()
                    .expect("test_name has an invalid extension");
                let extension = cap.get(1).unwrap().as_str();

                expected_file =
                    format!("{}{}{}", test_name.trim_end_matches(extension), suffix, extension,);
                from_file = test_name.to_string();
                assert!(test_names.next().is_none(), "two mir pass names specified for MIR dump");
                to_file = None;
                // the pass name is the third to last string in the test name
                // this gets pushed into passes
                passes.push(
                    test_name.split('.').rev().nth(2).expect("invalid test format").to_string(),
                );
            };
            if !expected_file.starts_with(&test_crate) {
                expected_file = format!("{}.{}", test_crate, expected_file);
            }
            let expected_file = test_dir.join(expected_file);

            out.push(MiroptTestFiles { expected_file, from_file, to_file, passes });
        }
    }

    out
}
