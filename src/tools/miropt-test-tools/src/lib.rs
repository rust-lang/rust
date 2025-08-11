use std::fs;
use std::path::Path;

pub struct MiroptTestFile {
    pub expected_file: std::path::PathBuf,
    pub from_file: String,
    pub to_file: Option<String>,
}

pub struct MiroptTest {
    pub run_filecheck: bool,
    pub suffix: String,
    pub files: Vec<MiroptTestFile>,
    /// Vec of passes under test to be dumped
    pub passes: Vec<String>,
}

pub enum PanicStrategy {
    Unwind,
    Abort,
}

fn output_file_suffix(testfile: &Path, bit_width: u32, panic_strategy: PanicStrategy) -> String {
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
        suffix.push_str(&format!(".{bit_width}bit"));
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
) -> MiroptTest {
    let mut out = Vec::new();
    let test_file_contents = fs::read_to_string(testfile).unwrap();

    let test_dir = testfile.parent().unwrap();
    let test_crate = testfile.file_stem().unwrap().to_str().unwrap().replace('-', "_");

    let suffix = output_file_suffix(testfile, bit_width, panic_strategy);
    let mut run_filecheck = true;
    let mut passes = Vec::new();

    for l in test_file_contents.lines() {
        if l.starts_with("// skip-filecheck") {
            run_filecheck = false;
            continue;
        }
        if l.starts_with("// EMIT_MIR ") {
            let test_name = l.trim_start_matches("// EMIT_MIR ").trim();
            let mut test_names = test_name.split(' ');
            // sometimes we specify two files so that we get a diff between the two files
            let test_name = test_names.next().unwrap();
            let mut expected_file;
            let from_file;
            let to_file;

            if test_name.ends_with(".diff") {
                let trimmed = test_name.trim_end_matches(".diff");
                passes.push(trimmed.split('.').next_back().unwrap().to_owned());
                let test_against = format!("{trimmed}.after.mir");
                from_file = format!("{trimmed}.before.mir");
                expected_file = format!("{trimmed}{suffix}.diff");
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

                expected_file = format!("{test_name}{suffix}.{first_pass}-{second_pass}.diff");
                let second_file = format!("{test_name}.{second_pass}.mir");
                from_file = format!("{test_name}.{first_pass}.mir");
                to_file = Some(second_file);
            } else {
                // Allow-list for file extensions that can be produced by MIR dumps.
                // Other extensions can be added here, as needed by new dump flags.
                static ALLOWED_EXT: &[&str] = &["mir", "dot"];
                let Some((test_name_wo_ext, test_name_ext)) = test_name.rsplit_once('.') else {
                    panic!(
                        "in {testfile:?}:\nEMIT_MIR has an unrecognized extension: {test_name}, expected one of {ALLOWED_EXT:?}"
                    )
                };
                if !ALLOWED_EXT.contains(&test_name_ext) {
                    panic!(
                        "in {testfile:?}:\nEMIT_MIR has an unrecognized extension: {test_name}, expected one of {ALLOWED_EXT:?}"
                    )
                }

                expected_file = format!("{test_name_wo_ext}{suffix}.{test_name_ext}");
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
                expected_file = format!("{test_crate}.{expected_file}");
            }
            let expected_file = test_dir.join(expected_file);

            out.push(MiroptTestFile { expected_file, from_file, to_file });
        }
        if !run_filecheck && l.trim_start().starts_with("// CHECK") {
            panic!("error: test contains filecheck directive but is marked `skip-filecheck`");
        }
    }

    MiroptTest { run_filecheck, suffix, files: out, passes }
}
