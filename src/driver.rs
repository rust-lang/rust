// error-pattern:yummy
#![feature(box_syntax)]
#![feature(rustc_private)]
#![allow(unknown_lints, missing_docs_in_private_items)]

use rustc_driver::{self, driver::CompileController, Compilation};
use rustc_plugin;
use std::process::{exit, Command};

#[allow(print_stdout)]
fn show_version() {
    println!(env!("CARGO_PKG_VERSION"));
}

pub fn main() {
    use std::env;

    if std::env::args().any(|a| a == "--version" || a == "-V") {
        show_version();
        return;
    }

    let sys_root = option_env!("SYSROOT")
        .map(String::from)
        .or_else(|| std::env::var("SYSROOT").ok())
        .or_else(|| {
            let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
            let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
            home.and_then(|home| toolchain.map(|toolchain| format!("{}/toolchains/{}", home, toolchain)))
        })
        .or_else(|| {
            Command::new("rustc")
                .arg("--print")
                .arg("sysroot")
                .output()
                .ok()
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_owned())
        })
        .expect("need to specify SYSROOT env var during clippy compilation, or use rustup or multirust");

    // Setting RUSTC_WRAPPER causes Cargo to pass 'rustc' as the first argument.
    // We're invoking the compiler programmatically, so we ignore this/
    let mut orig_args: Vec<String> = env::args().collect();
    if orig_args.len() <= 1 {
        std::process::exit(1);
    }
    if orig_args[1] == "rustc" {
        // we still want to be able to invoke it normally though
        orig_args.remove(1);
    }
    // this conditional check for the --sysroot flag is there so users can call
    // `clippy_driver` directly
    // without having to pass --sysroot or anything
    let mut args: Vec<String> = if orig_args.iter().any(|s| s == "--sysroot") {
        orig_args.clone()
    } else {
        orig_args
            .clone()
            .into_iter()
            .chain(Some("--sysroot".to_owned()))
            .chain(Some(sys_root))
            .collect()
    };

    // this check ensures that dependencies are built but not linted and the final
    // crate is
    // linted but not built
    let clippy_enabled = env::var("CLIPPY_TESTS").ok().map_or(false, |val| val == "true")
        || orig_args.iter().any(|s| s == "--emit=dep-info,metadata");

    if clippy_enabled {
        args.extend_from_slice(&["--cfg".to_owned(), r#"feature="cargo-clippy""#.to_owned()]);
        if let Ok(extra_args) = env::var("CLIPPY_ARGS") {
            args.extend(
                extra_args
                    .split("__CLIPPY_HACKERY__")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
            );
        }
    }

    let mut controller = CompileController::basic();
    if clippy_enabled {
        controller.after_parse.callback = Box::new(move |state| {
            let mut registry = rustc_plugin::registry::Registry::new(
                state.session,
                state
                    .krate
                    .as_ref()
                    .expect(
                        "at this compilation stage \
                         the crate must be parsed",
                    )
                    .span,
            );
            registry.args_hidden = Some(Vec::new());
            clippy_lints::register_plugins(&mut registry);

            let rustc_plugin::registry::Registry {
                early_lint_passes,
                late_lint_passes,
                lint_groups,
                llvm_passes,
                attributes,
                ..
            } = registry;
            let sess = &state.session;
            let mut ls = sess.lint_store.borrow_mut();
            for pass in early_lint_passes {
                ls.register_early_pass(Some(sess), true, pass);
            }
            for pass in late_lint_passes {
                ls.register_late_pass(Some(sess), true, pass);
            }

            for (name, to) in lint_groups {
                ls.register_group(Some(sess), true, name, to);
            }
            clippy_lints::register_pre_expansion_lints(sess, &mut ls);

            sess.plugin_llvm_passes.borrow_mut().extend(llvm_passes);
            sess.plugin_attributes.borrow_mut().extend(attributes);
        });
    }
    controller.compilation_done.stop = Compilation::Stop;

    if rustc_driver::run_compiler(&args, Box::new(controller), None, None)
        .0
        .is_err()
    {
        exit(101);
    }
}
