#![feature(rustc_private)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
#[allow(unused_extern_crates)]
extern crate rustc_driver;
#[allow(unused_extern_crates)]
extern crate rustc_interface;
#[allow(unused_extern_crates)]
extern crate rustc_plugin;

use rustc_interface::interface;
use rustc_tools_util::*;

use std::path::Path;
use std::process::{exit, Command};

mod lintlist;

/// If a command-line option matches `find_arg`, then apply the predicate `pred` on its value. If
/// true, then return it. The parameter is assumed to be either `--arg=value` or `--arg value`.
fn arg_value<'a>(
    args: impl IntoIterator<Item = &'a String>,
    find_arg: &str,
    pred: impl Fn(&str) -> bool,
) -> Option<&'a str> {
    let mut args = args.into_iter().map(String::as_str);

    while let Some(arg) = args.next() {
        let arg: Vec<_> = arg.splitn(2, '=').collect();
        if arg.get(0) != Some(&find_arg) {
            continue;
        }

        let value = arg.get(1).cloned().or_else(|| args.next());
        if value.as_ref().map_or(false, |p| pred(p)) {
            return value;
        }
    }
    None
}

#[test]
fn test_arg_value() {
    let args: Vec<_> = ["--bar=bar", "--foobar", "123", "--foo"]
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    assert_eq!(arg_value(None, "--foobar", |_| true), None);
    assert_eq!(arg_value(&args, "--bar", |_| false), None);
    assert_eq!(arg_value(&args, "--bar", |_| true), Some("bar"));
    assert_eq!(arg_value(&args, "--bar", |p| p == "bar"), Some("bar"));
    assert_eq!(arg_value(&args, "--bar", |p| p == "foo"), None);
    assert_eq!(arg_value(&args, "--foobar", |p| p == "foo"), None);
    assert_eq!(arg_value(&args, "--foobar", |p| p == "123"), Some("123"));
    assert_eq!(arg_value(&args, "--foo", |_| true), None);
}

#[allow(clippy::too_many_lines)]

struct ClippyCallbacks;

impl rustc_driver::Callbacks for ClippyCallbacks {
    fn after_parsing(&mut self, compiler: &interface::Compiler) -> bool {
        let sess = compiler.session();
        let mut registry = rustc_plugin::registry::Registry::new(
            sess,
            compiler
                .parse()
                .expect(
                    "at this compilation stage \
                     the crate must be parsed",
                )
                .peek()
                .span,
        );
        registry.args_hidden = Some(Vec::new());

        let conf = clippy_lints::read_conf(&registry);
        clippy_lints::register_plugins(&mut registry, &conf);

        let rustc_plugin::registry::Registry {
            early_lint_passes,
            late_lint_passes,
            lint_groups,
            llvm_passes,
            attributes,
            ..
        } = registry;
        let mut ls = sess.lint_store.borrow_mut();
        for pass in early_lint_passes {
            ls.register_early_pass(Some(sess), true, false, pass);
        }
        for pass in late_lint_passes {
            ls.register_late_pass(Some(sess), true, false, false, pass);
        }

        for (name, (to, deprecated_name)) in lint_groups {
            ls.register_group(Some(sess), true, name, deprecated_name, to);
        }
        clippy_lints::register_pre_expansion_lints(sess, &mut ls, &conf);
        clippy_lints::register_renamed(&mut ls);

        sess.plugin_llvm_passes.borrow_mut().extend(llvm_passes);
        sess.plugin_attributes.borrow_mut().extend(attributes);

        // Continue execution
        true
    }
}

pub fn main() {
    rustc_driver::init_rustc_env_logger();
    exit(
        rustc_driver::report_ices_to_stderr_if_any(move || {
            use std::env;

            if std::env::args().any(|a| a == "--version" || a == "-V") {
                let version_info = rustc_tools_util::get_version_info!();
                println!("{}", version_info);
                exit(0);
            }

            if std::env::args().any(|a| a == "--help" || a == "-h") {
                println!(
                    "\
Checks a package to catch common mistakes and improve your Rust code.

Usage:
    cargo clippy [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
    -V, --version            Print version info and exit

Other options are the same as `cargo check`.

To allow or deny a lint from the command line you can use `cargo clippy --`
with:

    -W --warn OPT       Set lint warnings
    -A --allow OPT      Set lint allowed
    -D --deny OPT       Set lint denied
    -F --forbid OPT     Set lint forbidden

You can use tool lints to allow or deny lints from your code, eg.:

    #[allow(clippy::needless_lifetimes)]
"
                );

                for lint in &lintlist::ALL_LINTS[..] {
                    println!("clippy::{},", lint.name);
                }
                exit(0);
            }

            let mut orig_args: Vec<String> = env::args().collect();

            // Get the sysroot, looking from most specific to this invocation to the least:
            // - command line
            // - runtime environment
            //    - SYSROOT
            //    - RUSTUP_HOME, MULTIRUST_HOME, RUSTUP_TOOLCHAIN, MULTIRUST_TOOLCHAIN
            // - sysroot from rustc in the path
            // - compile-time environment
            let sys_root_arg = arg_value(&orig_args, "--sysroot", |_| true);
            let have_sys_root_arg = sys_root_arg.is_some();
            let sys_root = sys_root_arg
                .map(std::string::ToString::to_string)
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
                .or_else(|| option_env!("SYSROOT").map(String::from))
                .expect("need to specify SYSROOT env var during clippy compilation, or use rustup or multirust");

            // Setting RUSTC_WRAPPER causes Cargo to pass 'rustc' as the first argument.
            // We're invoking the compiler programmatically, so we ignore this/
            if orig_args.len() <= 1 {
                std::process::exit(1);
            }
            if Path::new(&orig_args[1]).file_stem() == Some("rustc".as_ref()) {
                // we still want to be able to invoke it normally though
                orig_args.remove(1);
            }
            // this conditional check for the --sysroot flag is there so users can call
            // `clippy_driver` directly
            // without having to pass --sysroot or anything
            let mut args: Vec<String> = if have_sys_root_arg {
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
                || arg_value(&orig_args, "--emit", |val| val.split(',').any(|e| e == "metadata")).is_some();

            if clippy_enabled {
                args.extend_from_slice(&["--cfg".to_owned(), r#"feature="cargo-clippy""#.to_owned()]);
                if let Ok(extra_args) = env::var("CLIPPY_ARGS") {
                    args.extend(extra_args.split("__CLIPPY_HACKERY__").filter_map(|s| {
                        if s.is_empty() {
                            None
                        } else {
                            Some(s.to_string())
                        }
                    }));
                }
            }

            let mut clippy = ClippyCallbacks;
            let mut default = rustc_driver::DefaultCallbacks;
            let callbacks: &mut (dyn rustc_driver::Callbacks + Send) =
                if clippy_enabled { &mut clippy } else { &mut default };
            let args = args;
            rustc_driver::run_compiler(&args, callbacks, None, None)
        })
        .and_then(|result| result)
        .is_err() as i32,
    )
}
