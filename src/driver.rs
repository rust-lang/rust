#![feature(rustc_private)]
#![feature(once_cell)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]
// warn on rustc internal lints
#![deny(rustc::internal)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_interface;
extern crate rustc_middle;

use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;
use rustc_tools_util::VersionInfo;

use std::borrow::Cow;
use std::env;
use std::lazy::SyncLazy;
use std::ops::Deref;
use std::panic;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

mod lintlist;

/// If a command-line option matches `find_arg`, then apply the predicate `pred` on its value. If
/// true, then return it. The parameter is assumed to be either `--arg=value` or `--arg value`.
fn arg_value<'a, T: Deref<Target = str>>(
    args: &'a [T],
    find_arg: &str,
    pred: impl Fn(&str) -> bool,
) -> Option<&'a str> {
    let mut args = args.iter().map(Deref::deref);
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

#[test]
fn test_arg_value() {
    let args = &["--bar=bar", "--foobar", "123", "--foo"];

    assert_eq!(arg_value(&[] as &[&str], "--foobar", |_| true), None);
    assert_eq!(arg_value(args, "--bar", |_| false), None);
    assert_eq!(arg_value(args, "--bar", |_| true), Some("bar"));
    assert_eq!(arg_value(args, "--bar", |p| p == "bar"), Some("bar"));
    assert_eq!(arg_value(args, "--bar", |p| p == "foo"), None);
    assert_eq!(arg_value(args, "--foobar", |p| p == "foo"), None);
    assert_eq!(arg_value(args, "--foobar", |p| p == "123"), Some("123"));
    assert_eq!(arg_value(args, "--foo", |_| true), None);
}

struct DefaultCallbacks;
impl rustc_driver::Callbacks for DefaultCallbacks {}

struct ClippyCallbacks;
impl rustc_driver::Callbacks for ClippyCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        let previous = config.register_lints.take();
        config.register_lints = Some(Box::new(move |sess, mut lint_store| {
            // technically we're ~guaranteed that this is none but might as well call anything that
            // is there already. Certainly it can't hurt.
            if let Some(previous) = &previous {
                (previous)(sess, lint_store);
            }

            let conf = clippy_lints::read_conf(&[], &sess);
            clippy_lints::register_plugins(&mut lint_store, &sess, &conf);
            clippy_lints::register_pre_expansion_lints(&mut lint_store);
            clippy_lints::register_renamed(&mut lint_store);
        }));

        // FIXME: #4825; This is required, because Clippy lints that are based on MIR have to be
        // run on the unoptimized MIR. On the other hand this results in some false negatives. If
        // MIR passes can be enabled / disabled separately, we should figure out, what passes to
        // use for Clippy.
        config.opts.debugging_opts.mir_opt_level = 0;
    }
}

#[allow(clippy::find_map, clippy::filter_map)]
fn describe_lints() {
    use lintlist::{Level, Lint, ALL_LINTS, LINT_LEVELS};
    use rustc_data_structures::fx::FxHashSet;

    println!(
        "
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> (deny <foo> and all attempts to override)

"
    );

    let lint_level = |lint: &Lint| {
        LINT_LEVELS
            .iter()
            .find(|level_mapping| level_mapping.0 == lint.group)
            .map(|(_, level)| match level {
                Level::Allow => "allow",
                Level::Warn => "warn",
                Level::Deny => "deny",
            })
            .unwrap()
    };

    let mut lints: Vec<_> = ALL_LINTS.iter().collect();
    // The sort doesn't case-fold but it's doubtful we care.
    lints.sort_by_cached_key(|x: &&Lint| (lint_level(x), x.name));

    let max_lint_name_len = lints
        .iter()
        .map(|lint| lint.name.len())
        .map(|len| len + "clippy::".len())
        .max()
        .unwrap_or(0);

    let padded = |x: &str| {
        let mut s = " ".repeat(max_lint_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    let scoped = |x: &str| format!("clippy::{}", x);

    let lint_groups: FxHashSet<_> = lints.iter().map(|lint| lint.group).collect();

    println!("Lint checks provided by clippy:\n");
    println!("    {}  {:7.7}  meaning", padded("name"), "default");
    println!("    {}  {:7.7}  -------", padded("----"), "-------");

    let print_lints = |lints: &[&Lint]| {
        for lint in lints {
            let name = lint.name.replace("_", "-");
            println!(
                "    {}  {:7.7}  {}",
                padded(&scoped(&name)),
                lint_level(lint),
                lint.desc
            );
        }
        println!("\n");
    };

    print_lints(&lints);

    let max_group_name_len = std::cmp::max(
        "clippy::all".len(),
        lint_groups
            .iter()
            .map(|group| group.len())
            .map(|len| len + "clippy::".len())
            .max()
            .unwrap_or(0),
    );

    let padded_group = |x: &str| {
        let mut s = " ".repeat(max_group_name_len - x.chars().count());
        s.push_str(x);
        s
    };

    println!("Lint groups provided by clippy:\n");
    println!("    {}  sub-lints", padded_group("name"));
    println!("    {}  ---------", padded_group("----"));
    println!("    {}  the set of all clippy lints", padded_group("clippy::all"));

    let print_lint_groups = || {
        for group in lint_groups {
            let name = group.to_lowercase().replace("_", "-");
            let desc = lints
                .iter()
                .filter(|&lint| lint.group == group)
                .map(|lint| lint.name)
                .map(|name| name.replace("_", "-"))
                .collect::<Vec<String>>()
                .join(", ");
            println!("    {}  {}", padded_group(&scoped(&name)), desc);
        }
        println!("\n");
    };

    print_lint_groups();
}

fn display_help() {
    println!(
        "\
Checks a package to catch common mistakes and improve your Rust code.

Usage:
    cargo clippy [options] [--] [<opts>...]

Common options:
    -h, --help               Print this message
        --rustc              Pass all args to rustc
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
}

const BUG_REPORT_URL: &str = "https://github.com/rust-lang/rust-clippy/issues/new";

static ICE_HOOK: SyncLazy<Box<dyn Fn(&panic::PanicInfo<'_>) + Sync + Send + 'static>> = SyncLazy::new(|| {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|info| report_clippy_ice(info, BUG_REPORT_URL)));
    hook
});

fn report_clippy_ice(info: &panic::PanicInfo<'_>, bug_report_url: &str) {
    // Invoke our ICE handler, which prints the actual panic message and optionally a backtrace
    (*ICE_HOOK)(info);

    // Separate the output with an empty line
    eprintln!();

    let emitter = Box::new(rustc_errors::emitter::EmitterWriter::stderr(
        rustc_errors::ColorConfig::Auto,
        None,
        false,
        false,
        None,
        false,
    ));
    let handler = rustc_errors::Handler::with_emitter(true, None, emitter);

    // a .span_bug or .bug call has already printed what
    // it wants to print.
    if !info.payload().is::<rustc_errors::ExplicitBug>() {
        let d = rustc_errors::Diagnostic::new(rustc_errors::Level::Bug, "unexpected panic");
        handler.emit_diagnostic(&d);
    }

    let version_info = rustc_tools_util::get_version_info!();

    let xs: Vec<Cow<'static, str>> = vec![
        "the compiler unexpectedly panicked. this is a bug.".into(),
        format!("we would appreciate a bug report: {}", bug_report_url).into(),
        format!("Clippy version: {}", version_info).into(),
    ];

    for note in &xs {
        handler.note_without_error(&note);
    }

    // If backtraces are enabled, also print the query stack
    let backtrace = env::var_os("RUST_BACKTRACE").map_or(false, |x| &x != "0");

    let num_frames = if backtrace { None } else { Some(2) };

    TyCtxt::try_print_query_stack(&handler, num_frames);
}

fn toolchain_path(home: Option<String>, toolchain: Option<String>) -> Option<PathBuf> {
    home.and_then(|home| {
        toolchain.map(|toolchain| {
            let mut path = PathBuf::from(home);
            path.push("toolchains");
            path.push(toolchain);
            path
        })
    })
}

pub fn main() {
    rustc_driver::init_rustc_env_logger();
    SyncLazy::force(&ICE_HOOK);
    exit(rustc_driver::catch_with_exit_code(move || {
        let mut orig_args: Vec<String> = env::args().collect();

        // Get the sysroot, looking from most specific to this invocation to the least:
        // - command line
        // - runtime environment
        //    - SYSROOT
        //    - RUSTUP_HOME, MULTIRUST_HOME, RUSTUP_TOOLCHAIN, MULTIRUST_TOOLCHAIN
        // - sysroot from rustc in the path
        // - compile-time environment
        //    - SYSROOT
        //    - RUSTUP_HOME, MULTIRUST_HOME, RUSTUP_TOOLCHAIN, MULTIRUST_TOOLCHAIN
        let sys_root_arg = arg_value(&orig_args, "--sysroot", |_| true);
        let have_sys_root_arg = sys_root_arg.is_some();
        let sys_root = sys_root_arg
            .map(PathBuf::from)
            .or_else(|| std::env::var("SYSROOT").ok().map(PathBuf::from))
            .or_else(|| {
                let home = std::env::var("RUSTUP_HOME")
                    .or_else(|_| std::env::var("MULTIRUST_HOME"))
                    .ok();
                let toolchain = std::env::var("RUSTUP_TOOLCHAIN")
                    .or_else(|_| std::env::var("MULTIRUST_TOOLCHAIN"))
                    .ok();
                toolchain_path(home, toolchain)
            })
            .or_else(|| {
                Command::new("rustc")
                    .arg("--print")
                    .arg("sysroot")
                    .output()
                    .ok()
                    .and_then(|out| String::from_utf8(out.stdout).ok())
                    .map(|s| PathBuf::from(s.trim()))
            })
            .or_else(|| option_env!("SYSROOT").map(PathBuf::from))
            .or_else(|| {
                let home = option_env!("RUSTUP_HOME")
                    .or(option_env!("MULTIRUST_HOME"))
                    .map(ToString::to_string);
                let toolchain = option_env!("RUSTUP_TOOLCHAIN")
                    .or(option_env!("MULTIRUST_TOOLCHAIN"))
                    .map(ToString::to_string);
                toolchain_path(home, toolchain)
            })
            .map(|pb| pb.to_string_lossy().to_string())
            .expect("need to specify SYSROOT env var during clippy compilation, or use rustup or multirust");

        // make "clippy-driver --rustc" work like a subcommand that passes further args to "rustc"
        // for example `clippy-driver --rustc --version` will print the rustc version that clippy-driver
        // uses
        if let Some(pos) = orig_args.iter().position(|arg| arg == "--rustc") {
            orig_args.remove(pos);
            orig_args[0] = "rustc".to_string();

            // if we call "rustc", we need to pass --sysroot here as well
            let mut args: Vec<String> = orig_args.clone();
            if !have_sys_root_arg {
                args.extend(vec!["--sysroot".into(), sys_root]);
            };

            return rustc_driver::RunCompiler::new(&args, &mut DefaultCallbacks).run();
        }

        if orig_args.iter().any(|a| a == "--version" || a == "-V") {
            let version_info = rustc_tools_util::get_version_info!();
            println!("{}", version_info);
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

        let should_describe_lints = || {
            let args: Vec<_> = env::args().collect();
            args.windows(2)
                .any(|args| args[1] == "help" && matches!(args[0].as_str(), "-W" | "-A" | "-D" | "-F"))
        };

        if !wrapper_mode && should_describe_lints() {
            describe_lints();
            exit(0);
        }

        // this conditional check for the --sysroot flag is there so users can call
        // `clippy_driver` directly
        // without having to pass --sysroot or anything
        let mut args: Vec<String> = orig_args.clone();
        if !have_sys_root_arg {
            args.extend(vec!["--sysroot".into(), sys_root]);
        };

        // this check ensures that dependencies are built but not linted and the final
        // crate is linted but not built
        let clippy_enabled = env::var("CLIPPY_TESTS").map_or(false, |val| val == "true")
            || arg_value(&orig_args, "--cap-lints", |val| val == "allow").is_none();

        if clippy_enabled {
            args.extend(vec!["--cfg".into(), r#"feature="cargo-clippy""#.into()]);
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
        let mut default = DefaultCallbacks;
        let callbacks: &mut (dyn rustc_driver::Callbacks + Send) =
            if clippy_enabled { &mut clippy } else { &mut default };
        rustc_driver::RunCompiler::new(&args, callbacks).run()
    }))
}
