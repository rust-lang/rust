#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![feature(rustc_private)]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]
// warn on rustc internal lints
#![warn(rustc::internal)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use clippy_utils::sym;
use rustc_interface::interface;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::ErrorOutputType;
use rustc_session::parse::ParseSess;
use rustc_span::symbol::Symbol;

use std::env;
use std::fs::read_to_string;
use std::path::Path;
use std::process::exit;

use anstream::println;

/// If a command-line option matches `find_arg`, then apply the predicate `pred` on its value. If
/// true, then return it. The parameter is assumed to be either `--arg=value` or `--arg value`.
fn arg_value<'a>(args: &'a [String], find_arg: &str, pred: impl Fn(&str) -> bool) -> Option<&'a str> {
    let mut args = args.iter().map(String::as_str);
    while let Some(arg) = args.next() {
        let mut arg = arg.splitn(2, '=');
        if arg.next() != Some(find_arg) {
            continue;
        }

        match arg.next().or_else(|| args.next()) {
            Some(v) if pred(v) => return Some(v),
            _ => {},
        }
    }
    None
}

fn has_arg(args: &[String], find_arg: &str) -> bool {
    args.iter().any(|arg| find_arg == arg.split('=').next().unwrap())
}

#[test]
fn test_arg_value() {
    let args = &["--bar=bar", "--foobar", "123", "--foo"].map(String::from);

    assert_eq!(arg_value(&[], "--foobar", |_| true), None);
    assert_eq!(arg_value(args, "--bar", |_| false), None);
    assert_eq!(arg_value(args, "--bar", |_| true), Some("bar"));
    assert_eq!(arg_value(args, "--bar", |p| p == "bar"), Some("bar"));
    assert_eq!(arg_value(args, "--bar", |p| p == "foo"), None);
    assert_eq!(arg_value(args, "--foobar", |p| p == "foo"), None);
    assert_eq!(arg_value(args, "--foobar", |p| p == "123"), Some("123"));
    assert_eq!(arg_value(args, "--foobar", |p| p.contains("12")), Some("123"));
    assert_eq!(arg_value(args, "--foo", |_| true), None);
}

#[test]
fn test_has_arg() {
    let args = &["--foo=bar", "-vV", "--baz"].map(String::from);
    assert!(has_arg(args, "--foo"));
    assert!(has_arg(args, "--baz"));
    assert!(has_arg(args, "-vV"));

    assert!(!has_arg(args, "--bar"));
}

fn track_clippy_args(psess: &mut ParseSess, args_env_var: Option<&str>) {
    psess
        .env_depinfo
        .get_mut()
        .insert((sym::CLIPPY_ARGS, args_env_var.map(Symbol::intern)));
}

/// Track files that may be accessed at runtime in `file_depinfo` so that cargo will re-run clippy
/// when any of them are modified
fn track_files(psess: &mut ParseSess) {
    let file_depinfo = psess.file_depinfo.get_mut();

    // Used by `clippy::cargo` lints and to determine the MSRV. `cargo clippy` executes `clippy-driver`
    // with the current directory set to `CARGO_MANIFEST_DIR` so a relative path is fine
    if Path::new("Cargo.toml").exists() {
        file_depinfo.insert(sym::Cargo_toml);
    }

    // `clippy.toml` will be automatically tracked as it's loaded with `sess.source_map().load_file()`

    // During development track the `clippy-driver` executable so that cargo will re-run clippy whenever
    // it is rebuilt
    #[expect(
        clippy::collapsible_if,
        reason = "Due to a bug in let_chains this if statement can't be collapsed"
    )]
    if cfg!(debug_assertions) {
        if let Ok(current_exe) = env::current_exe()
            && let Some(current_exe) = current_exe.to_str()
        {
            file_depinfo.insert(Symbol::intern(current_exe));
        }
    }
}

struct DefaultCallbacks;
impl rustc_driver::Callbacks for DefaultCallbacks {}

/// This is different from `DefaultCallbacks` that it will inform Cargo to track the value of
/// `CLIPPY_ARGS` environment variable.
struct RustcCallbacks {
    clippy_args_var: Option<String>,
}

impl rustc_driver::Callbacks for RustcCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        let clippy_args_var = self.clippy_args_var.take();
        config.psess_created = Some(Box::new(move |psess| {
            track_clippy_args(psess, clippy_args_var.as_deref());
        }));
    }
}

struct ClippyCallbacks {
    clippy_args_var: Option<String>,
}

impl rustc_driver::Callbacks for ClippyCallbacks {
    // JUSTIFICATION: necessary in clippy driver to set `mir_opt_level`
    #[allow(rustc::bad_opt_access)]
    fn config(&mut self, config: &mut interface::Config) {
        let conf_path = clippy_config::lookup_conf_file();
        let previous = config.register_lints.take();
        let clippy_args_var = self.clippy_args_var.take();
        config.psess_created = Some(Box::new(move |psess| {
            track_clippy_args(psess, clippy_args_var.as_deref());
            track_files(psess);

            // Trigger a rebuild if CLIPPY_CONF_DIR changes. The value must be a valid string so
            // changes between dirs that are invalid UTF-8 will not trigger rebuilds
            psess.env_depinfo.get_mut().insert((
                sym::CLIPPY_CONF_DIR,
                env::var("CLIPPY_CONF_DIR").ok().map(|dir| Symbol::intern(&dir)),
            ));
        }));
        config.register_lints = Some(Box::new(move |sess, lint_store| {
            // technically we're ~guaranteed that this is none but might as well call anything that
            // is there already. Certainly it can't hurt.
            if let Some(previous) = &previous {
                (previous)(sess, lint_store);
            }

            let conf = clippy_config::Conf::read(sess, &conf_path);
            clippy_lints::register_lints(lint_store, conf);
            #[cfg(feature = "internal")]
            clippy_lints_internal::register_lints(lint_store);
        }));
        config.extra_symbols = sym::EXTRA_SYMBOLS.into();

        // FIXME: #4825; This is required, because Clippy lints that are based on MIR have to be
        // run on the unoptimized MIR. On the other hand this results in some false negatives. If
        // MIR passes can be enabled / disabled separately, we should figure out, what passes to
        // use for Clippy.
        config.opts.unstable_opts.mir_opt_level = Some(0);
        config.opts.unstable_opts.mir_enable_passes =
            vec![("CheckNull".to_owned(), false), ("CheckAlignment".to_owned(), false)];

        // Disable flattening and inlining of format_args!(), so the HIR matches with the AST.
        config.opts.unstable_opts.flatten_format_args = false;
    }
}

#[allow(clippy::ignored_unit_patterns)]
fn display_help() {
    println!("{}", help_message());
}

const BUG_REPORT_URL: &str = "https://github.com/rust-lang/rust-clippy/issues/new?template=ice.yml";

#[allow(clippy::too_many_lines)]
#[allow(clippy::ignored_unit_patterns)]
pub fn main() {
    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    rustc_driver::init_rustc_env_logger(&early_dcx);

    rustc_driver::install_ice_hook(BUG_REPORT_URL, |dcx| {
        // FIXME: this macro calls unwrap internally but is called in a panicking context!  It's not
        // as simple as moving the call from the hook to main, because `install_ice_hook` doesn't
        // accept a generic closure.
        let version_info = rustc_tools_util::get_version_info!();
        dcx.handle().note(format!("Clippy version: {version_info}"));
    });

    exit(rustc_driver::catch_with_exit_code(move || {
        let mut orig_args = rustc_driver::args::raw_args(&early_dcx);

        let has_sysroot_arg = |args: &mut [String]| -> bool {
            if has_arg(args, "--sysroot") {
                return true;
            }
            // https://doc.rust-lang.org/rustc/command-line-arguments.html#path-load-command-line-flags-from-a-path
            // Beside checking for existence of `--sysroot` on the command line, we need to
            // check for the arg files that are prefixed with @ as well to be consistent with rustc
            for arg in args.iter() {
                if let Some(arg_file_path) = arg.strip_prefix('@')
                    && let Ok(arg_file) = read_to_string(arg_file_path)
                {
                    let split_arg_file: Vec<String> = arg_file.lines().map(ToString::to_string).collect();
                    if has_arg(&split_arg_file, "--sysroot") {
                        return true;
                    }
                }
            }
            false
        };

        let sys_root_env = std::env::var("SYSROOT").ok();
        let pass_sysroot_env_if_given = |args: &mut Vec<String>, sys_root_env| {
            if let Some(sys_root) = sys_root_env
                && !has_sysroot_arg(args)
            {
                args.extend(vec!["--sysroot".into(), sys_root]);
            }
        };

        // make "clippy-driver --rustc" work like a subcommand that passes further args to "rustc"
        // for example `clippy-driver --rustc --version` will print the rustc version that clippy-driver
        // uses
        if let Some(pos) = orig_args.iter().position(|arg| arg == "--rustc") {
            orig_args.remove(pos);
            orig_args[0] = "rustc".to_string();

            let mut args: Vec<String> = orig_args.clone();
            pass_sysroot_env_if_given(&mut args, sys_root_env);

            rustc_driver::run_compiler(&args, &mut DefaultCallbacks);
            return;
        }

        if orig_args.iter().any(|a| a == "--version" || a == "-V") {
            let version_info = rustc_tools_util::get_version_info!();

            println!("{version_info}");
            exit(0);
        }

        // Setting RUSTC_WRAPPER causes Cargo to pass 'rustc' as the first argument.
        // We're invoking the compiler programmatically, so we ignore this/
        let wrapper_mode = orig_args.get(1).map(Path::new).and_then(Path::file_stem) == Some("rustc".as_ref());

        if wrapper_mode {
            // we still want to be able to invoke it normally though
            orig_args.remove(1);
        }

        if !wrapper_mode && (orig_args.iter().any(|a| a == "--help" || a == "-h") || orig_args.len() == 1) {
            display_help();
            exit(0);
        }

        let mut args: Vec<String> = orig_args.clone();
        pass_sysroot_env_if_given(&mut args, sys_root_env);

        let mut no_deps = false;
        let clippy_args_var = env::var("CLIPPY_ARGS").ok();
        let clippy_args = clippy_args_var
            .as_deref()
            .unwrap_or_default()
            .split("__CLIPPY_HACKERY__")
            .filter_map(|s| match s {
                "" => None,
                "--no-deps" => {
                    no_deps = true;
                    None
                },
                _ => Some(s.to_string()),
            })
            .chain(vec!["--cfg".into(), "clippy".into()])
            .collect::<Vec<String>>();

        // If no Clippy lints will be run we do not need to run Clippy
        let cap_lints_allow = arg_value(&orig_args, "--cap-lints", |val| val == "allow").is_some()
            && arg_value(&orig_args, "--force-warn", |val| val.contains("clippy::")).is_none();

        // If `--no-deps` is enabled only lint the primary package
        let relevant_package = !no_deps || env::var("CARGO_PRIMARY_PACKAGE").is_ok();

        // Do not run Clippy for Cargo's info queries so that invalid CLIPPY_ARGS are not cached
        // https://github.com/rust-lang/cargo/issues/14385
        let info_query = has_arg(&orig_args, "-vV") || has_arg(&orig_args, "--print");

        let clippy_enabled = !cap_lints_allow && relevant_package && !info_query;
        if clippy_enabled {
            args.extend(clippy_args);
            rustc_driver::run_compiler(&args, &mut ClippyCallbacks { clippy_args_var });
        } else {
            rustc_driver::run_compiler(&args, &mut RustcCallbacks { clippy_args_var });
        }
    }))
}

#[must_use]
fn help_message() -> &'static str {
    color_print::cstr!(
        "Checks a file to catch common mistakes and improve your Rust code.
Run <cyan>clippy-driver</> with the same arguments you use for <cyan>rustc</>

<green,bold>Usage</>:
    <cyan,bold>clippy-driver</> <cyan>[OPTIONS] INPUT</>

<green,bold>Common options:</>
    <cyan,bold>-h</>, <cyan,bold>--help</>               Print this message
    <cyan,bold>-V</>, <cyan,bold>--version</>            Print version info and exit
    <cyan,bold>--rustc</>                  Pass all arguments to <cyan>rustc</>

<green,bold>Allowing / Denying lints</>
You can use tool lints to allow or deny lints from your code, e.g.:

    <yellow,bold>#[allow(clippy::needless_lifetimes)]</>
"
    )
}
