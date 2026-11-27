//@ needs-target-std
//@ ignore-cross-compile
//@ ignore-windows-gnu
// GNU Linker for Windows is non-deterministic. (from `reproducible-build-2` test in this suite)

use std::rc::Rc;

use run_make_support::{rfs, run_in_tmpdir, rustc};

/// Test that parallel compiler produces identical metadata for the
/// `DefId`s synthesized on demand by `coroutine_by_move_body_def_id`
/// (`DefKind::SyntheticCoroutineBody`), across many async closures that
/// each need one.
fn main() {
    const FILE_NAME: &str = "by-move-body";
    let rmeta_name = format!("{FILE_NAME}.rmeta");

    let mut reference = None;
    let mut reference_stderr = None;

    for _ in 0..10 {
        // Tmp dir as previous runs affect output binary on windows.
        run_in_tmpdir(|| {
            let mut rustc = rustc();
            rustc
                .input(format!("{FILE_NAME}.rs"))
                .arg("--edition=2021")
                .arg("-Zremap-cwd-prefix=reproducible_dir")
                .arg("-Ccodegen-units=1")
                .arg("-Zthreads=2")
                .arg("--crate-type=lib")
                .emit("metadata")
                .output(&rmeta_name);

            let current_stderr = rustc.run().stderr_utf8();

            let current = Rc::new(rfs::read(&rmeta_name));
            reference.get_or_insert(Rc::clone(&current));
            let reference_stderr = reference_stderr.get_or_insert_with(|| current_stderr.clone());

            if Some(current.clone()) != reference {
                let reference_bytes = reference.as_ref().unwrap();
                let (pos, (left_byte, right_byte)) = current
                    .iter()
                    .zip(reference_bytes.iter())
                    .enumerate()
                    .find(|(_, (c, r))| c != r)
                    .unwrap();
                let range_start = pos.saturating_sub(1);
                let range_end = (pos + 3).min(current.len()).min(reference_bytes.len());
                panic!(
                    "left: {current:x?}\nright: {reference:x?}\n \
                     differs at byte {pos}: left = {left_byte:#x}, right = {right_byte:#x}\n\
                     left range [{range_start}..{range_end}]: {:x?}\n\
                     right range [{range_start}..{range_end}]: {:x?}\n\
                     left stderr:\n{current_stderr}\n\
                     right stderr:\n{reference_stderr}",
                    &current[range_start..range_end],
                    &reference_bytes[range_start..range_end],
                )
            }
            assert_eq!(Some(current), reference);
        });
    }
}
