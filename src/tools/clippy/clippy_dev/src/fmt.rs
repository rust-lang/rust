use crate::generate::gen_sorted_lints_file;
use crate::new_parse_cx;
use crate::utils::{
    ErrAction, FileUpdater, UpdateMode, UpdateStatus, VecBuf, expect_action, run_with_output, split_args_for_threads,
    walk_dir_no_dot_or_target,
};
use core::fmt::Write as _;
use std::io::Read as _;
use std::process::{self, Command, Stdio};

/// Format the symbols list
fn fmt_syms(update_mode: UpdateMode) {
    FileUpdater::default().update_file_checked(
        "cargo dev fmt",
        update_mode,
        "clippy_utils/src/sym.rs",
        &mut |_, text: &str, new_text: &mut String| {
            let (pre, conf) = text.split_once("generate! {\n").expect("can't find generate! call");
            let (conf, post) = conf.split_once("\n}\n").expect("can't find end of generate! call");
            let mut lines = conf
                .lines()
                .map(|line| {
                    let line = line.trim();
                    line.strip_suffix(',').unwrap_or(line).trim_end()
                })
                .collect::<Vec<_>>();
            lines.sort_unstable();
            write!(
                new_text,
                "{pre}generate! {{\n    {},\n}}\n{post}",
                lines.join(",\n    "),
            )
            .unwrap();
            if text == new_text {
                UpdateStatus::Unchanged
            } else {
                UpdateStatus::Changed
            }
        },
    );
}

fn run_rustfmt(update_mode: UpdateMode) {
    let mut rustfmt_path = String::from_utf8(run_with_output(
        "rustup which rustfmt",
        Command::new("rustup").args(["which", "rustfmt"]),
    ))
    .expect("invalid rustfmt path");
    rustfmt_path.truncate(rustfmt_path.trim_end().len());

    let args: Vec<_> = walk_dir_no_dot_or_target(".")
        .filter_map(|e| {
            let e = expect_action(e, ErrAction::Read, ".");
            e.path()
                .as_os_str()
                .as_encoded_bytes()
                .ends_with(b".rs")
                .then(|| e.into_path().into_os_string())
        })
        .collect();

    let mut children: Vec<_> = split_args_for_threads(
        32,
        || {
            let mut cmd = Command::new(&rustfmt_path);
            if update_mode.is_check() {
                cmd.arg("--check");
            }
            cmd.stdout(Stdio::null())
                .stdin(Stdio::null())
                .stderr(Stdio::piped())
                .args(["--unstable-features", "--skip-children"]);
            cmd
        },
        args.iter(),
    )
    .map(|mut cmd| expect_action(cmd.spawn(), ErrAction::Run, "rustfmt"))
    .collect();

    for child in &mut children {
        let status = expect_action(child.wait(), ErrAction::Run, "rustfmt");
        match (update_mode, status.exit_ok()) {
            (UpdateMode::Check | UpdateMode::Change, Ok(())) => {},
            (UpdateMode::Check, Err(_)) => {
                let mut s = String::new();
                if let Some(mut stderr) = child.stderr.take()
                    && stderr.read_to_string(&mut s).is_ok()
                {
                    eprintln!("{s}");
                }
                eprintln!("Formatting check failed!\nRun `cargo dev fmt` to update.");
                process::exit(1);
            },
            (UpdateMode::Change, e) => {
                let mut s = String::new();
                if let Some(mut stderr) = child.stderr.take()
                    && stderr.read_to_string(&mut s).is_ok()
                {
                    eprintln!("{s}");
                }
                expect_action(e, ErrAction::Run, "rustfmt");
            },
        }
    }
}

// the "main" function of cargo dev fmt
pub fn run(update_mode: UpdateMode) {
    fmt_syms(update_mode);
    new_parse_cx(|cx| {
        let mut lint_data = cx.parse_lint_decls();
        let mut conf_data = cx.parse_conf_mac();
        cx.dcx.exit_on_err();

        let mut updater = FileUpdater::default();

        #[expect(clippy::mutable_key_type)]
        let mut lints = lint_data.mk_file_to_lint_decl_map();
        let mut ranges = VecBuf::with_capacity(256);
        for passes in lint_data.iter_passes_by_file_mut() {
            let file = passes[0].decl_sp.file;
            let mut lints = lints.remove(file);
            let lints = lints.as_deref_mut().unwrap_or_default();
            updater.update_loaded_file_checked("cargo dev fmt", update_mode, file, &mut |_, src, dst| {
                gen_sorted_lints_file(src, dst, lints, passes, &mut ranges);
                UpdateStatus::from_changed(src != dst)
            });
        }
        for (&file, lints) in &mut lints {
            updater.update_loaded_file_checked("cargo dev fmt", update_mode, file, &mut |_, src, dst| {
                gen_sorted_lints_file(src, dst, lints, &mut [], &mut ranges);
                UpdateStatus::from_changed(src != dst)
            });
        }

        updater.update_loaded_file_checked(
            "cargo dev fmt",
            update_mode,
            conf_data.decl_sp.file,
            &mut |_, src, dst| {
                dst.push_str(&src[..conf_data.decl_sp.range.start as usize]);
                conf_data.gen_mac(src, dst);
                dst.push_str(&src[conf_data.decl_sp.range.end as usize..]);
                UpdateStatus::from_changed(src != dst)
            },
        );
    });

    run_rustfmt(update_mode);
}
