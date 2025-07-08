use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::{env, io, iter, mem, str};

use cc::windows_registry;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_metadata::{
    find_native_static_library, try_find_native_dynamic_library, try_find_native_static_library,
};
use rustc_middle::bug;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::middle::exported_symbols;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo, SymbolExportKind};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{self, CrateType, DebugInfo, LinkerPluginLto, Lto, OptLevel, Strip};
use rustc_span::sym;
use rustc_target::spec::{Cc, LinkOutputKind, LinkerFlavor, Lld};
use tracing::{debug, warn};

use super::command::Command;
use super::symbol_export;
use crate::errors;

#[cfg(test)]
mod tests;

/// Disables non-English messages from localized linkers.
/// Such messages may cause issues with text encoding on Windows (#35785)
/// and prevent inspection of linker output in case of errors, which we occasionally do.
/// This should be acceptable because other messages from rustc are in English anyway,
/// and may also be desirable to improve searchability of the linker diagnostics.
pub(crate) fn disable_localization(linker: &mut Command) {
    // No harm in setting both env vars simultaneously.
    // Unix-style linkers.
    linker.env("LC_ALL", "C");
    // MSVC's `link.exe`.
    linker.env("VSLANG", "1033");
}

/// The third parameter is for env vars, used on windows to set up the
/// path for MSVC to find its DLLs, and gcc to find its bundled
/// toolchain
pub(crate) fn get_linker<'a>(
    sess: &'a Session,
    linker: &Path,
    flavor: LinkerFlavor,
    self_contained: bool,
    target_cpu: &'a str,
) -> Box<dyn Linker + 'a> {
    let msvc_tool = windows_registry::find_tool(&sess.target.arch, "link.exe");

    // If our linker looks like a batch script on Windows then to execute this
    // we'll need to spawn `cmd` explicitly. This is primarily done to handle
    // emscripten where the linker is `emcc.bat` and needs to be spawned as
    // `cmd /c emcc.bat ...`.
    //
    // This worked historically but is needed manually since #42436 (regression
    // was tagged as #42791) and some more info can be found on #44443 for
    // emscripten itself.
    let mut cmd = match linker.to_str() {
        Some(linker) if cfg!(windows) && linker.ends_with(".bat") => Command::bat_script(linker),
        _ => match flavor {
            LinkerFlavor::Gnu(Cc::No, Lld::Yes)
            | LinkerFlavor::Darwin(Cc::No, Lld::Yes)
            | LinkerFlavor::WasmLld(Cc::No)
            | LinkerFlavor::Msvc(Lld::Yes) => Command::lld(linker, flavor.lld_flavor()),
            LinkerFlavor::Msvc(Lld::No)
                if sess.opts.cg.linker.is_none() && sess.target.linker.is_none() =>
            {
                Command::new(msvc_tool.as_ref().map_or(linker, |t| t.path()))
            }
            _ => Command::new(linker),
        },
    };

    // UWP apps have API restrictions enforced during Store submissions.
    // To comply with the Windows App Certification Kit,
    // MSVC needs to link with the Store versions of the runtime libraries (vcruntime, msvcrt, etc).
    let t = &sess.target;
    if matches!(flavor, LinkerFlavor::Msvc(..)) && t.vendor == "uwp" {
        if let Some(ref tool) = msvc_tool {
            let original_path = tool.path();
            if let Some(root_lib_path) = original_path.ancestors().nth(4) {
                let arch = match t.arch.as_ref() {
                    "x86_64" => Some("x64"),
                    "x86" => Some("x86"),
                    "aarch64" => Some("arm64"),
                    "arm" => Some("arm"),
                    _ => None,
                };
                if let Some(ref a) = arch {
                    // FIXME: Move this to `fn linker_with_args`.
                    let mut arg = OsString::from("/LIBPATH:");
                    arg.push(format!("{}\\lib\\{}\\store", root_lib_path.display(), a));
                    cmd.arg(&arg);
                } else {
                    warn!("arch is not supported");
                }
            } else {
                warn!("MSVC root path lib location not found");
            }
        } else {
            warn!("link.exe not found");
        }
    }

    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = sess.get_tools_search_paths(self_contained);
    let mut msvc_changed_path = false;
    if sess.target.is_like_msvc
        && let Some(ref tool) = msvc_tool
    {
        cmd.args(tool.args());
        for (k, v) in tool.env() {
            if k == "PATH" {
                new_path.extend(env::split_paths(v));
                msvc_changed_path = true;
            } else {
                cmd.env(k, v);
            }
        }
    }

    if !msvc_changed_path && let Some(path) = env::var_os("PATH") {
        new_path.extend(env::split_paths(&path));
    }
    cmd.env("PATH", env::join_paths(new_path).unwrap());

    // FIXME: Move `/LIBPATH` addition for uwp targets from the linker construction
    // to the linker args construction.
    assert!(cmd.get_args().is_empty() || sess.target.vendor == "uwp");
    match flavor {
        LinkerFlavor::Unix(Cc::No) if sess.target.os == "l4re" => {
            Box::new(L4Bender::new(cmd, sess)) as Box<dyn Linker>
        }
        LinkerFlavor::Unix(Cc::No) if sess.target.os == "aix" => {
            Box::new(AixLinker::new(cmd, sess)) as Box<dyn Linker>
        }
        LinkerFlavor::WasmLld(Cc::No) => Box::new(WasmLd::new(cmd, sess)) as Box<dyn Linker>,
        LinkerFlavor::Gnu(cc, _)
        | LinkerFlavor::Darwin(cc, _)
        | LinkerFlavor::WasmLld(cc)
        | LinkerFlavor::Unix(cc) => Box::new(GccLinker {
            cmd,
            sess,
            target_cpu,
            hinted_static: None,
            is_ld: cc == Cc::No,
            is_gnu: flavor.is_gnu(),
            uses_lld: flavor.uses_lld(),
        }) as Box<dyn Linker>,
        LinkerFlavor::Msvc(..) => Box::new(MsvcLinker { cmd, sess }) as Box<dyn Linker>,
        LinkerFlavor::EmCc => Box::new(EmLinker { cmd, sess }) as Box<dyn Linker>,
        LinkerFlavor::Bpf => Box::new(BpfLinker { cmd, sess }) as Box<dyn Linker>,
        LinkerFlavor::Llbc => Box::new(LlbcLinker { cmd, sess }) as Box<dyn Linker>,
        LinkerFlavor::Ptx => Box::new(PtxLinker { cmd, sess }) as Box<dyn Linker>,
    }
}

// Note: Ideally neither these helper function, nor the macro-generated inherent methods below
// would exist, and these functions would live in `trait Linker`.
// Unfortunately, adding these functions to `trait Linker` make it `dyn`-incompatible.
// If the methods are added to the trait with `where Self: Sized` bounds, then even a separate
// implementation of them for `dyn Linker {}` wouldn't work due to a conflict with those
// uncallable methods in the trait.

/// Just pass the arguments to the linker as is.
/// It is assumed that they are correctly prepared in advance.
fn verbatim_args<L: Linker + ?Sized>(
    l: &mut L,
    args: impl IntoIterator<Item: AsRef<OsStr>>,
) -> &mut L {
    for arg in args {
        l.cmd().arg(arg);
    }
    l
}
/// Add underlying linker arguments to C compiler command, by wrapping them in
/// `-Wl` or `-Xlinker`.
fn convert_link_args_to_cc_args(cmd: &mut Command, args: impl IntoIterator<Item: AsRef<OsStr>>) {
    let mut combined_arg = OsString::from("-Wl");
    for arg in args {
        // If the argument itself contains a comma, we need to emit it
        // as `-Xlinker`, otherwise we can use `-Wl`.
        if arg.as_ref().as_encoded_bytes().contains(&b',') {
            // Emit current `-Wl` argument, if any has been built.
            if combined_arg != OsStr::new("-Wl") {
                cmd.arg(combined_arg);
                // Begin next `-Wl` argument.
                combined_arg = OsString::from("-Wl");
            }

            // Emit `-Xlinker` argument.
            cmd.arg("-Xlinker");
            cmd.arg(arg);
        } else {
            // Append to `-Wl` argument.
            combined_arg.push(",");
            combined_arg.push(arg);
        }
    }
    // Emit final `-Wl` argument.
    if combined_arg != OsStr::new("-Wl") {
        cmd.arg(combined_arg);
    }
}
/// Arguments for the underlying linker.
/// Add options to pass them through cc wrapper if `Linker` is a cc wrapper.
fn link_args<L: Linker + ?Sized>(l: &mut L, args: impl IntoIterator<Item: AsRef<OsStr>>) -> &mut L {
    if !l.is_cc() {
        verbatim_args(l, args);
    } else {
        convert_link_args_to_cc_args(l.cmd(), args);
    }
    l
}
/// Arguments for the cc wrapper specifically.
/// Check that it's indeed a cc wrapper and pass verbatim.
fn cc_args<L: Linker + ?Sized>(l: &mut L, args: impl IntoIterator<Item: AsRef<OsStr>>) -> &mut L {
    assert!(l.is_cc());
    verbatim_args(l, args)
}
/// Arguments supported by both underlying linker and cc wrapper, pass verbatim.
fn link_or_cc_args<L: Linker + ?Sized>(
    l: &mut L,
    args: impl IntoIterator<Item: AsRef<OsStr>>,
) -> &mut L {
    verbatim_args(l, args)
}

macro_rules! generate_arg_methods {
    ($($ty:ty)*) => { $(
        impl $ty {
            #[allow(unused)]
            pub(crate) fn verbatim_args(&mut self, args: impl IntoIterator<Item: AsRef<OsStr>>) -> &mut Self {
                verbatim_args(self, args)
            }
            #[allow(unused)]
            pub(crate) fn verbatim_arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
                verbatim_args(self, iter::once(arg))
            }
            #[allow(unused)]
            pub(crate) fn link_args(&mut self, args: impl IntoIterator<Item: AsRef<OsStr>>) -> &mut Self {
                link_args(self, args)
            }
            #[allow(unused)]
            pub(crate) fn link_arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
                link_args(self, iter::once(arg))
            }
            #[allow(unused)]
            pub(crate) fn cc_args(&mut self, args: impl IntoIterator<Item: AsRef<OsStr>>) -> &mut Self {
                cc_args(self, args)
            }
            #[allow(unused)]
            pub(crate) fn cc_arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
                cc_args(self, iter::once(arg))
            }
            #[allow(unused)]
            pub(crate) fn link_or_cc_args(&mut self, args: impl IntoIterator<Item: AsRef<OsStr>>) -> &mut Self {
                link_or_cc_args(self, args)
            }
            #[allow(unused)]
            pub(crate) fn link_or_cc_arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
                link_or_cc_args(self, iter::once(arg))
            }
        }
    )* }
}

generate_arg_methods! {
    GccLinker<'_>
    MsvcLinker<'_>
    EmLinker<'_>
    WasmLd<'_>
    L4Bender<'_>
    AixLinker<'_>
    LlbcLinker<'_>
    PtxLinker<'_>
    BpfLinker<'_>
    dyn Linker + '_
}

/// Linker abstraction used by `back::link` to build up the command to invoke a
/// linker.
///
/// This trait is the total list of requirements needed by `back::link` and
/// represents the meaning of each option being passed down. This trait is then
/// used to dispatch on whether a GNU-like linker (generally `ld.exe`) or an
/// MSVC linker (e.g., `link.exe`) is being used.
pub(crate) trait Linker {
    fn cmd(&mut self) -> &mut Command;
    fn is_cc(&self) -> bool {
        false
    }
    fn set_output_kind(
        &mut self,
        output_kind: LinkOutputKind,
        crate_type: CrateType,
        out_filename: &Path,
    );
    fn link_dylib_by_name(&mut self, _name: &str, _verbatim: bool, _as_needed: bool) {
        bug!("dylib linked with unsupported linker")
    }
    fn link_dylib_by_path(&mut self, _path: &Path, _as_needed: bool) {
        bug!("dylib linked with unsupported linker")
    }
    fn link_framework_by_name(&mut self, _name: &str, _verbatim: bool, _as_needed: bool) {
        bug!("framework linked with unsupported linker")
    }
    fn link_staticlib_by_name(&mut self, name: &str, verbatim: bool, whole_archive: bool);
    fn link_staticlib_by_path(&mut self, path: &Path, whole_archive: bool);
    fn include_path(&mut self, path: &Path) {
        link_or_cc_args(link_or_cc_args(self, &["-L"]), &[path]);
    }
    fn framework_path(&mut self, _path: &Path) {
        bug!("framework path set with unsupported linker")
    }
    fn output_filename(&mut self, path: &Path) {
        link_or_cc_args(link_or_cc_args(self, &["-o"]), &[path]);
    }
    fn add_object(&mut self, path: &Path) {
        link_or_cc_args(self, &[path]);
    }
    fn gc_sections(&mut self, keep_metadata: bool);
    fn no_gc_sections(&mut self);
    fn full_relro(&mut self);
    fn partial_relro(&mut self);
    fn no_relro(&mut self);
    fn optimize(&mut self);
    fn pgo_gen(&mut self);
    fn control_flow_guard(&mut self);
    fn ehcont_guard(&mut self);
    fn debuginfo(&mut self, strip: Strip, natvis_debugger_visualizers: &[PathBuf]);
    fn no_crt_objects(&mut self);
    fn no_default_libraries(&mut self);
    fn export_symbols(
        &mut self,
        tmpdir: &Path,
        crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    );
    fn subsystem(&mut self, subsystem: &str);
    fn linker_plugin_lto(&mut self);
    fn add_eh_frame_header(&mut self) {}
    fn add_no_exec(&mut self) {}
    fn add_as_needed(&mut self) {}
    fn reset_per_library_state(&mut self) {}
}

impl dyn Linker + '_ {
    pub(crate) fn take_cmd(&mut self) -> Command {
        mem::replace(self.cmd(), Command::new(""))
    }
}

struct GccLinker<'a> {
    cmd: Command,
    sess: &'a Session,
    target_cpu: &'a str,
    hinted_static: Option<bool>, // Keeps track of the current hinting mode.
    // Link as ld
    is_ld: bool,
    is_gnu: bool,
    uses_lld: bool,
}

impl<'a> GccLinker<'a> {
    fn takes_hints(&self) -> bool {
        // Really this function only returns true if the underlying linker
        // configured for a compiler is binutils `ld.bfd` and `ld.gold`. We
        // don't really have a foolproof way to detect that, so rule out some
        // platforms where currently this is guaranteed to *not* be the case:
        //
        // * On OSX they have their own linker, not binutils'
        // * For WebAssembly the only functional linker is LLD, which doesn't
        //   support hint flags
        !self.sess.target.is_like_darwin && !self.sess.target.is_like_wasm
    }

    // Some platforms take hints about whether a library is static or dynamic.
    // For those that support this, we ensure we pass the option if the library
    // was flagged "static" (most defaults are dynamic) to ensure that if
    // libfoo.a and libfoo.so both exist that the right one is chosen.
    fn hint_static(&mut self) {
        if !self.takes_hints() {
            return;
        }
        if self.hinted_static != Some(true) {
            self.link_arg("-Bstatic");
            self.hinted_static = Some(true);
        }
    }

    fn hint_dynamic(&mut self) {
        if !self.takes_hints() {
            return;
        }
        if self.hinted_static != Some(false) {
            self.link_arg("-Bdynamic");
            self.hinted_static = Some(false);
        }
    }

    fn push_linker_plugin_lto_args(&mut self, plugin_path: Option<&OsStr>) {
        if let Some(plugin_path) = plugin_path {
            let mut arg = OsString::from("-plugin=");
            arg.push(plugin_path);
            self.link_arg(&arg);
        }

        let opt_level = match self.sess.opts.optimize {
            config::OptLevel::No => "O0",
            config::OptLevel::Less => "O1",
            config::OptLevel::More | config::OptLevel::Size | config::OptLevel::SizeMin => "O2",
            config::OptLevel::Aggressive => "O3",
        };

        if let Some(path) = &self.sess.opts.unstable_opts.profile_sample_use {
            self.link_arg(&format!("-plugin-opt=sample-profile={}", path.display()));
        };
        self.link_args(&[
            &format!("-plugin-opt={opt_level}"),
            &format!("-plugin-opt=mcpu={}", self.target_cpu),
        ]);
    }

    fn build_dylib(&mut self, crate_type: CrateType, out_filename: &Path) {
        // On mac we need to tell the linker to let this library be rpathed
        if self.sess.target.is_like_darwin {
            if self.is_cc() {
                // `-dynamiclib` makes `cc` pass `-dylib` to the linker.
                self.cc_arg("-dynamiclib");
            } else {
                self.link_arg("-dylib");
                // Clang also sets `-dynamic`, but that's implied by `-dylib`, so unnecessary.
            }

            // Note that the `osx_rpath_install_name` option here is a hack
            // purely to support bootstrap right now, we should get a more
            // principled solution at some point to force the compiler to pass
            // the right `-Wl,-install_name` with an `@rpath` in it.
            if self.sess.opts.cg.rpath || self.sess.opts.unstable_opts.osx_rpath_install_name {
                let mut rpath = OsString::from("@rpath/");
                rpath.push(out_filename.file_name().unwrap());
                self.link_arg("-install_name").link_arg(rpath);
            }
        } else {
            self.link_or_cc_arg("-shared");
            if let Some(name) = out_filename.file_name() {
                if self.sess.target.is_like_windows {
                    // The output filename already contains `dll_suffix` so
                    // the resulting import library will have a name in the
                    // form of libfoo.dll.a
                    let (prefix, suffix) = self.sess.staticlib_components(false);
                    let mut implib_name = OsString::from(prefix);
                    implib_name.push(name);
                    implib_name.push(suffix);
                    let mut out_implib = OsString::from("--out-implib=");
                    out_implib.push(out_filename.with_file_name(implib_name));
                    self.link_arg(out_implib);
                } else if crate_type == CrateType::Dylib {
                    // When dylibs are linked by a full path this value will get into `DT_NEEDED`
                    // instead of the full path, so the library can be later found in some other
                    // location than that specific path.
                    let mut soname = OsString::from("-soname=");
                    soname.push(name);
                    self.link_arg(soname);
                }
            }
        }
    }

    fn with_as_needed(&mut self, as_needed: bool, f: impl FnOnce(&mut Self)) {
        if !as_needed {
            if self.sess.target.is_like_darwin {
                // FIXME(81490): ld64 doesn't support these flags but macOS 11
                // has -needed-l{} / -needed_library {}
                // but we have no way to detect that here.
                self.sess.dcx().emit_warn(errors::Ld64UnimplementedModifier);
            } else if self.is_gnu && !self.sess.target.is_like_windows {
                self.link_arg("--no-as-needed");
            } else {
                self.sess.dcx().emit_warn(errors::LinkerUnsupportedModifier);
            }
        }

        f(self);

        if !as_needed {
            if self.sess.target.is_like_darwin {
                // See above FIXME comment
            } else if self.is_gnu && !self.sess.target.is_like_windows {
                self.link_arg("--as-needed");
            }
        }
    }
}

impl<'a> Linker for GccLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn is_cc(&self) -> bool {
        !self.is_ld
    }

    fn set_output_kind(
        &mut self,
        output_kind: LinkOutputKind,
        crate_type: CrateType,
        out_filename: &Path,
    ) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe => {
                if !self.is_ld && self.is_gnu {
                    self.cc_arg("-no-pie");
                }
            }
            LinkOutputKind::DynamicPicExe => {
                // noop on windows w/ gcc & ld, error w/ lld
                if !self.sess.target.is_like_windows {
                    // `-pie` works for both gcc wrapper and ld.
                    self.link_or_cc_arg("-pie");
                }
            }
            LinkOutputKind::StaticNoPicExe => {
                // `-static` works for both gcc wrapper and ld.
                self.link_or_cc_arg("-static");
                if !self.is_ld && self.is_gnu {
                    self.cc_arg("-no-pie");
                }
            }
            LinkOutputKind::StaticPicExe => {
                if !self.is_ld {
                    // Note that combination `-static -pie` doesn't work as expected
                    // for the gcc wrapper, `-static` in that case suppresses `-pie`.
                    self.cc_arg("-static-pie");
                } else {
                    // `--no-dynamic-linker` and `-z text` are not strictly necessary for producing
                    // a static pie, but currently passed because gcc and clang pass them.
                    // The former suppresses the `INTERP` ELF header specifying dynamic linker,
                    // which is otherwise implicitly injected by ld (but not lld).
                    // The latter doesn't change anything, only ensures that everything is pic.
                    self.link_args(&["-static", "-pie", "--no-dynamic-linker", "-z", "text"]);
                }
            }
            LinkOutputKind::DynamicDylib => self.build_dylib(crate_type, out_filename),
            LinkOutputKind::StaticDylib => {
                self.link_or_cc_arg("-static");
                self.build_dylib(crate_type, out_filename);
            }
            LinkOutputKind::WasiReactorExe => {
                self.link_args(&["--entry", "_initialize"]);
            }
        }

        // VxWorks compiler driver introduced `--static-crt` flag specifically for rustc,
        // it switches linking for libc and similar system libraries to static without using
        // any `#[link]` attributes in the `libc` crate, see #72782 for details.
        // FIXME: Switch to using `#[link]` attributes in the `libc` crate
        // similarly to other targets.
        if self.sess.target.os == "vxworks"
            && matches!(
                output_kind,
                LinkOutputKind::StaticNoPicExe
                    | LinkOutputKind::StaticPicExe
                    | LinkOutputKind::StaticDylib
            )
        {
            self.cc_arg("--static-crt");
        }

        // avr-none doesn't have default ISA, users must specify which specific
        // CPU (well, microcontroller) they are targetting using `-Ctarget-cpu`.
        //
        // Currently this makes sense only when using avr-gcc as a linker, since
        // it brings a couple of hand-written important intrinsics from libgcc.
        if self.sess.target.arch == "avr" && !self.uses_lld {
            self.verbatim_arg(format!("-mmcu={}", self.target_cpu));
        }
    }

    fn link_dylib_by_name(&mut self, name: &str, verbatim: bool, as_needed: bool) {
        if self.sess.target.os == "illumos" && name == "c" {
            // libc will be added via late_link_args on illumos so that it will
            // appear last in the library search order.
            // FIXME: This should be replaced by a more complete and generic
            // mechanism for controlling the order of library arguments passed
            // to the linker.
            return;
        }
        self.hint_dynamic();
        self.with_as_needed(as_needed, |this| {
            let colon = if verbatim && this.is_gnu { ":" } else { "" };
            this.link_or_cc_arg(format!("-l{colon}{name}"));
        });
    }

    fn link_dylib_by_path(&mut self, path: &Path, as_needed: bool) {
        self.hint_dynamic();
        self.with_as_needed(as_needed, |this| {
            this.link_or_cc_arg(path);
        })
    }

    fn link_framework_by_name(&mut self, name: &str, _verbatim: bool, as_needed: bool) {
        self.hint_dynamic();
        if !as_needed {
            // FIXME(81490): ld64 as of macOS 11 supports the -needed_framework
            // flag but we have no way to detect that here.
            // self.link_or_cc_arg("-needed_framework").link_or_cc_arg(name);
            self.sess.dcx().emit_warn(errors::Ld64UnimplementedModifier);
        }
        self.link_or_cc_args(&["-framework", name]);
    }

    fn link_staticlib_by_name(&mut self, name: &str, verbatim: bool, whole_archive: bool) {
        self.hint_static();
        let colon = if verbatim && self.is_gnu { ":" } else { "" };
        if !whole_archive {
            self.link_or_cc_arg(format!("-l{colon}{name}"));
        } else if self.sess.target.is_like_darwin {
            // -force_load is the macOS equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            self.link_arg("-force_load");
            self.link_arg(find_native_static_library(name, verbatim, self.sess));
        } else {
            self.link_arg("--whole-archive")
                .link_or_cc_arg(format!("-l{colon}{name}"))
                .link_arg("--no-whole-archive");
        }
    }

    fn link_staticlib_by_path(&mut self, path: &Path, whole_archive: bool) {
        self.hint_static();
        if !whole_archive {
            self.link_or_cc_arg(path);
        } else if self.sess.target.is_like_darwin {
            self.link_arg("-force_load").link_arg(path);
        } else {
            self.link_arg("--whole-archive").link_arg(path).link_arg("--no-whole-archive");
        }
    }

    fn framework_path(&mut self, path: &Path) {
        self.link_or_cc_arg("-F").link_or_cc_arg(path);
    }
    fn full_relro(&mut self) {
        self.link_args(&["-z", "relro", "-z", "now"]);
    }
    fn partial_relro(&mut self) {
        self.link_args(&["-z", "relro"]);
    }
    fn no_relro(&mut self) {
        self.link_args(&["-z", "norelro"]);
    }

    fn gc_sections(&mut self, keep_metadata: bool) {
        // The dead_strip option to the linker specifies that functions and data
        // unreachable by the entry point will be removed. This is quite useful
        // with Rust's compilation model of compiling libraries at a time into
        // one object file. For example, this brings hello world from 1.7MB to
        // 458K.
        //
        // Note that this is done for both executables and dynamic libraries. We
        // won't get much benefit from dylibs because LLVM will have already
        // stripped away as much as it could. This has not been seen to impact
        // link times negatively.
        //
        // -dead_strip can't be part of the pre_link_args because it's also used
        // for partial linking when using multiple codegen units (-r). So we
        // insert it here.
        if self.sess.target.is_like_darwin {
            self.link_arg("-dead_strip");

        // If we're building a dylib, we don't use --gc-sections because LLVM
        // has already done the best it can do, and we also don't want to
        // eliminate the metadata. If we're building an executable, however,
        // --gc-sections drops the size of hello world from 1.8MB to 597K, a 67%
        // reduction.
        } else if (self.is_gnu || self.sess.target.is_like_wasm) && !keep_metadata {
            self.link_arg("--gc-sections");
        }
    }

    fn no_gc_sections(&mut self) {
        if self.is_gnu || self.sess.target.is_like_wasm {
            self.link_arg("--no-gc-sections");
        }
    }

    fn optimize(&mut self) {
        if !self.is_gnu && !self.sess.target.is_like_wasm {
            return;
        }

        // GNU-style linkers support optimization with -O. GNU ld doesn't
        // need a numeric argument, but other linkers do.
        if self.sess.opts.optimize == config::OptLevel::More
            || self.sess.opts.optimize == config::OptLevel::Aggressive
        {
            self.link_arg("-O1");
        }
    }

    fn pgo_gen(&mut self) {
        if !self.is_gnu {
            return;
        }

        // If we're doing PGO generation stuff and on a GNU-like linker, use the
        // "-u" flag to properly pull in the profiler runtime bits.
        //
        // This is because LLVM otherwise won't add the needed initialization
        // for us on Linux (though the extra flag should be harmless if it
        // does).
        //
        // See https://reviews.llvm.org/D14033 and https://reviews.llvm.org/D14030.
        //
        // Though it may be worth to try to revert those changes upstream, since
        // the overhead of the initialization should be minor.
        self.link_or_cc_args(&["-u", "__llvm_profile_runtime"]);
    }

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn debuginfo(&mut self, strip: Strip, _: &[PathBuf]) {
        // MacOS linker doesn't support stripping symbols directly anymore.
        if self.sess.target.is_like_darwin {
            return;
        }

        match strip {
            Strip::None => {}
            Strip::Debuginfo => {
                // The illumos linker does not support --strip-debug although
                // it does support --strip-all as a compatibility alias for -s.
                // The --strip-debug case is handled by running an external
                // `strip` utility as a separate step after linking.
                if !self.sess.target.is_like_solaris {
                    self.link_arg("--strip-debug");
                }
            }
            Strip::Symbols => {
                self.link_arg("--strip-all");
            }
        }
        match self.sess.opts.unstable_opts.debuginfo_compression {
            config::DebugInfoCompression::None => {}
            config::DebugInfoCompression::Zlib => {
                self.link_arg("--compress-debug-sections=zlib");
            }
            config::DebugInfoCompression::Zstd => {
                self.link_arg("--compress-debug-sections=zstd");
            }
        }
    }

    fn no_crt_objects(&mut self) {
        if !self.is_ld {
            self.cc_arg("-nostartfiles");
        }
    }

    fn no_default_libraries(&mut self) {
        if !self.is_ld {
            self.cc_arg("-nodefaultlibs");
        }
    }

    fn export_symbols(
        &mut self,
        tmpdir: &Path,
        crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        // Symbol visibility in object files typically takes care of this.
        if crate_type == CrateType::Executable {
            let should_export_executable_symbols =
                self.sess.opts.unstable_opts.export_executable_symbols;
            if self.sess.target.override_export_symbols.is_none()
                && !should_export_executable_symbols
            {
                return;
            }
        }

        // We manually create a list of exported symbols to ensure we don't expose any more.
        // The object files have far more public symbols than we actually want to export,
        // so we hide them all here.

        if !self.sess.target.limit_rdylib_exports {
            return;
        }

        let is_windows = self.sess.target.is_like_windows;
        let path = tmpdir.join(if is_windows { "list.def" } else { "list" });

        debug!("EXPORTED SYMBOLS:");

        if self.sess.target.is_like_darwin {
            // Write a plain, newline-separated list of symbols
            let res: io::Result<()> = try {
                let mut f = File::create_buffered(&path)?;
                for (sym, _) in symbols {
                    debug!("  _{sym}");
                    writeln!(f, "_{sym}")?;
                }
            };
            if let Err(error) = res {
                self.sess.dcx().emit_fatal(errors::LibDefWriteFailure { error });
            }
        } else if is_windows {
            let res: io::Result<()> = try {
                let mut f = File::create_buffered(&path)?;

                // .def file similar to MSVC one but without LIBRARY section
                // because LD doesn't like when it's empty
                writeln!(f, "EXPORTS")?;
                for (symbol, kind) in symbols {
                    let kind_marker = if *kind == SymbolExportKind::Data { " DATA" } else { "" };
                    debug!("  _{symbol}");
                    // Quote the name in case it's reserved by linker in some way
                    // (this accounts for names with dots in particular).
                    writeln!(f, "  \"{symbol}\"{kind_marker}")?;
                }
            };
            if let Err(error) = res {
                self.sess.dcx().emit_fatal(errors::LibDefWriteFailure { error });
            }
        } else {
            // Write an LD version script
            let res: io::Result<()> = try {
                let mut f = File::create_buffered(&path)?;
                writeln!(f, "{{")?;
                if !symbols.is_empty() {
                    writeln!(f, "  global:")?;
                    for (sym, _) in symbols {
                        debug!("    {sym};");
                        writeln!(f, "    {sym};")?;
                    }
                }
                writeln!(f, "\n  local:\n    *;\n}};")?;
            };
            if let Err(error) = res {
                self.sess.dcx().emit_fatal(errors::VersionScriptWriteFailure { error });
            }
        }

        if self.sess.target.is_like_darwin {
            self.link_arg("-exported_symbols_list").link_arg(path);
        } else if self.sess.target.is_like_solaris {
            self.link_arg("-M").link_arg(path);
        } else if is_windows {
            self.link_arg(path);
        } else {
            let mut arg = OsString::from("--version-script=");
            arg.push(path);
            self.link_arg(arg).link_arg("--no-undefined-version");
        }
    }

    fn subsystem(&mut self, subsystem: &str) {
        self.link_args(&["--subsystem", subsystem]);
    }

    fn reset_per_library_state(&mut self) {
        self.hint_dynamic(); // Reset to default before returning the composed command line.
    }

    fn linker_plugin_lto(&mut self) {
        match self.sess.opts.cg.linker_plugin_lto {
            LinkerPluginLto::Disabled => {
                // Nothing to do
            }
            LinkerPluginLto::LinkerPluginAuto => {
                self.push_linker_plugin_lto_args(None);
            }
            LinkerPluginLto::LinkerPlugin(ref path) => {
                self.push_linker_plugin_lto_args(Some(path.as_os_str()));
            }
        }
    }

    // Add the `GNU_EH_FRAME` program header which is required to locate unwinding information.
    // Some versions of `gcc` add it implicitly, some (e.g. `musl-gcc`) don't,
    // so we just always add it.
    fn add_eh_frame_header(&mut self) {
        self.link_arg("--eh-frame-hdr");
    }

    fn add_no_exec(&mut self) {
        if self.sess.target.is_like_windows {
            self.link_arg("--nxcompat");
        } else if self.is_gnu {
            self.link_args(&["-z", "noexecstack"]);
        }
    }

    fn add_as_needed(&mut self) {
        if self.is_gnu && !self.sess.target.is_like_windows {
            self.link_arg("--as-needed");
        } else if self.sess.target.is_like_solaris {
            // -z ignore is the Solaris equivalent to the GNU ld --as-needed option
            self.link_args(&["-z", "ignore"]);
        }
    }
}

struct MsvcLinker<'a> {
    cmd: Command,
    sess: &'a Session,
}

impl<'a> Linker for MsvcLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        output_kind: LinkOutputKind,
        _crate_type: CrateType,
        out_filename: &Path,
    ) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe
            | LinkOutputKind::DynamicPicExe
            | LinkOutputKind::StaticNoPicExe
            | LinkOutputKind::StaticPicExe => {}
            LinkOutputKind::DynamicDylib | LinkOutputKind::StaticDylib => {
                self.link_arg("/DLL");
                let mut arg: OsString = "/IMPLIB:".into();
                arg.push(out_filename.with_extension("dll.lib"));
                self.link_arg(arg);
            }
            LinkOutputKind::WasiReactorExe => {
                panic!("can't link as reactor on non-wasi target");
            }
        }
    }

    fn link_dylib_by_name(&mut self, name: &str, verbatim: bool, _as_needed: bool) {
        // On MSVC-like targets rustc supports import libraries using alternative naming
        // scheme (`libfoo.a`) unsupported by linker, search for such libraries manually.
        if let Some(path) = try_find_native_dynamic_library(self.sess, name, verbatim) {
            self.link_arg(path);
        } else {
            self.link_arg(format!("{}{}", name, if verbatim { "" } else { ".lib" }));
        }
    }

    fn link_dylib_by_path(&mut self, path: &Path, _as_needed: bool) {
        // When producing a dll, MSVC linker may not emit an implib file if the dll doesn't export
        // any symbols, so we skip linking if the implib file is not present.
        let implib_path = path.with_extension("dll.lib");
        if implib_path.exists() {
            self.link_or_cc_arg(implib_path);
        }
    }

    fn link_staticlib_by_name(&mut self, name: &str, verbatim: bool, whole_archive: bool) {
        // On MSVC-like targets rustc supports static libraries using alternative naming
        // scheme (`libfoo.a`) unsupported by linker, search for such libraries manually.
        if let Some(path) = try_find_native_static_library(self.sess, name, verbatim) {
            self.link_staticlib_by_path(&path, whole_archive);
        } else {
            let opts = if whole_archive { "/WHOLEARCHIVE:" } else { "" };
            let (prefix, suffix) = self.sess.staticlib_components(verbatim);
            self.link_arg(format!("{opts}{prefix}{name}{suffix}"));
        }
    }

    fn link_staticlib_by_path(&mut self, path: &Path, whole_archive: bool) {
        if !whole_archive {
            self.link_arg(path);
        } else {
            let mut arg = OsString::from("/WHOLEARCHIVE:");
            arg.push(path);
            self.link_arg(arg);
        }
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        // MSVC's ICF (Identical COMDAT Folding) link optimization is
        // slow for Rust and thus we disable it by default when not in
        // optimization build.
        if self.sess.opts.optimize != config::OptLevel::No {
            self.link_arg("/OPT:REF,ICF");
        } else {
            // It is necessary to specify NOICF here, because /OPT:REF
            // implies ICF by default.
            self.link_arg("/OPT:REF,NOICF");
        }
    }

    fn no_gc_sections(&mut self) {
        self.link_arg("/OPT:NOREF,NOICF");
    }

    fn full_relro(&mut self) {
        // noop
    }

    fn partial_relro(&mut self) {
        // noop
    }

    fn no_relro(&mut self) {
        // noop
    }

    fn no_crt_objects(&mut self) {
        // noop
    }

    fn no_default_libraries(&mut self) {
        self.link_arg("/NODEFAULTLIB");
    }

    fn include_path(&mut self, path: &Path) {
        let mut arg = OsString::from("/LIBPATH:");
        arg.push(path);
        self.link_arg(&arg);
    }

    fn output_filename(&mut self, path: &Path) {
        let mut arg = OsString::from("/OUT:");
        arg.push(path);
        self.link_arg(&arg);
    }

    fn optimize(&mut self) {
        // Needs more investigation of `/OPT` arguments
    }

    fn pgo_gen(&mut self) {
        // Nothing needed here.
    }

    fn control_flow_guard(&mut self) {
        self.link_arg("/guard:cf");
    }

    fn ehcont_guard(&mut self) {
        if self.sess.target.pointer_width == 64 {
            self.link_arg("/guard:ehcont");
        }
    }

    fn debuginfo(&mut self, _strip: Strip, natvis_debugger_visualizers: &[PathBuf]) {
        // This will cause the Microsoft linker to generate a PDB file
        // from the CodeView line tables in the object files.
        self.link_arg("/DEBUG");

        // Default to emitting only the file name of the PDB file into
        // the binary instead of the full path. Emitting the full path
        // may leak private information (such as user names).
        // See https://github.com/rust-lang/rust/issues/87825.
        //
        // This default behavior can be overridden by explicitly passing
        // `-Clink-arg=/PDBALTPATH:...` to rustc.
        self.link_arg("/PDBALTPATH:%_PDB%");

        // This will cause the Microsoft linker to embed .natvis info into the PDB file
        let natvis_dir_path = self.sess.opts.sysroot.path().join("lib\\rustlib\\etc");
        if let Ok(natvis_dir) = fs::read_dir(&natvis_dir_path) {
            for entry in natvis_dir {
                match entry {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.extension() == Some("natvis".as_ref()) {
                            let mut arg = OsString::from("/NATVIS:");
                            arg.push(path);
                            self.link_arg(arg);
                        }
                    }
                    Err(error) => {
                        self.sess.dcx().emit_warn(errors::NoNatvisDirectory { error });
                    }
                }
            }
        }

        // This will cause the Microsoft linker to embed .natvis info for all crates into the PDB file
        for path in natvis_debugger_visualizers {
            let mut arg = OsString::from("/NATVIS:");
            arg.push(path);
            self.link_arg(arg);
        }
    }

    // Currently the compiler doesn't use `dllexport` (an LLVM attribute) to
    // export symbols from a dynamic library. When building a dynamic library,
    // however, we're going to want some symbols exported, so this function
    // generates a DEF file which lists all the symbols.
    //
    // The linker will read this `*.def` file and export all the symbols from
    // the dynamic library. Note that this is not as simple as just exporting
    // all the symbols in the current crate (as specified by `codegen.reachable`)
    // but rather we also need to possibly export the symbols of upstream
    // crates. Upstream rlibs may be linked statically to this dynamic library,
    // in which case they may continue to transitively be used and hence need
    // their symbols exported.
    fn export_symbols(
        &mut self,
        tmpdir: &Path,
        crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        // Symbol visibility takes care of this typically
        if crate_type == CrateType::Executable {
            let should_export_executable_symbols =
                self.sess.opts.unstable_opts.export_executable_symbols;
            if !should_export_executable_symbols {
                return;
            }
        }

        let path = tmpdir.join("lib.def");
        let res: io::Result<()> = try {
            let mut f = File::create_buffered(&path)?;

            // Start off with the standard module name header and then go
            // straight to exports.
            writeln!(f, "LIBRARY")?;
            writeln!(f, "EXPORTS")?;
            for (symbol, kind) in symbols {
                let kind_marker = if *kind == SymbolExportKind::Data { " DATA" } else { "" };
                debug!("  _{symbol}");
                writeln!(f, "  {symbol}{kind_marker}")?;
            }
        };
        if let Err(error) = res {
            self.sess.dcx().emit_fatal(errors::LibDefWriteFailure { error });
        }
        let mut arg = OsString::from("/DEF:");
        arg.push(path);
        self.link_arg(&arg);
    }

    fn subsystem(&mut self, subsystem: &str) {
        // Note that previous passes of the compiler validated this subsystem,
        // so we just blindly pass it to the linker.
        self.link_arg(&format!("/SUBSYSTEM:{subsystem}"));

        // Windows has two subsystems we're interested in right now, the console
        // and windows subsystems. These both implicitly have different entry
        // points (starting symbols). The console entry point starts with
        // `mainCRTStartup` and the windows entry point starts with
        // `WinMainCRTStartup`. These entry points, defined in system libraries,
        // will then later probe for either `main` or `WinMain`, respectively to
        // start the application.
        //
        // In Rust we just always generate a `main` function so we want control
        // to always start there, so we force the entry point on the windows
        // subsystem to be `mainCRTStartup` to get everything booted up
        // correctly.
        //
        // For more information see RFC #1665
        if subsystem == "windows" {
            self.link_arg("/ENTRY:mainCRTStartup");
        }
    }

    fn linker_plugin_lto(&mut self) {
        // Do nothing
    }

    fn add_no_exec(&mut self) {
        self.link_arg("/NXCOMPAT");
    }
}

struct EmLinker<'a> {
    cmd: Command,
    sess: &'a Session,
}

impl<'a> Linker for EmLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn is_cc(&self) -> bool {
        true
    }

    fn set_output_kind(
        &mut self,
        _output_kind: LinkOutputKind,
        _crate_type: CrateType,
        _out_filename: &Path,
    ) {
    }

    fn link_dylib_by_name(&mut self, name: &str, _verbatim: bool, _as_needed: bool) {
        // Emscripten always links statically
        self.link_or_cc_args(&["-l", name]);
    }

    fn link_dylib_by_path(&mut self, path: &Path, _as_needed: bool) {
        self.link_or_cc_arg(path);
    }

    fn link_staticlib_by_name(&mut self, name: &str, _verbatim: bool, _whole_archive: bool) {
        self.link_or_cc_args(&["-l", name]);
    }

    fn link_staticlib_by_path(&mut self, path: &Path, _whole_archive: bool) {
        self.link_or_cc_arg(path);
    }

    fn full_relro(&mut self) {
        // noop
    }

    fn partial_relro(&mut self) {
        // noop
    }

    fn no_relro(&mut self) {
        // noop
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        // noop
    }

    fn no_gc_sections(&mut self) {
        // noop
    }

    fn optimize(&mut self) {
        // Emscripten performs own optimizations
        self.cc_arg(match self.sess.opts.optimize {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::More => "-O2",
            OptLevel::Aggressive => "-O3",
            OptLevel::Size => "-Os",
            OptLevel::SizeMin => "-Oz",
        });
    }

    fn pgo_gen(&mut self) {
        // noop, but maybe we need something like the gnu linker?
    }

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn debuginfo(&mut self, _strip: Strip, _: &[PathBuf]) {
        // Preserve names or generate source maps depending on debug info
        // For more information see https://emscripten.org/docs/tools_reference/emcc.html#emcc-g
        self.cc_arg(match self.sess.opts.debuginfo {
            DebugInfo::None => "-g0",
            DebugInfo::Limited | DebugInfo::LineTablesOnly | DebugInfo::LineDirectivesOnly => {
                "--profiling-funcs"
            }
            DebugInfo::Full => "-g",
        });
    }

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {
        self.cc_arg("-nodefaultlibs");
    }

    fn export_symbols(
        &mut self,
        _tmpdir: &Path,
        _crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        debug!("EXPORTED SYMBOLS:");

        self.cc_arg("-s");

        let mut arg = OsString::from("EXPORTED_FUNCTIONS=");
        let encoded = serde_json::to_string(
            &symbols.iter().map(|(sym, _)| "_".to_owned() + sym).collect::<Vec<_>>(),
        )
        .unwrap();
        debug!("{encoded}");

        arg.push(encoded);

        self.cc_arg(arg);
    }

    fn subsystem(&mut self, _subsystem: &str) {
        // noop
    }

    fn linker_plugin_lto(&mut self) {
        // Do nothing
    }
}

struct WasmLd<'a> {
    cmd: Command,
    sess: &'a Session,
}

impl<'a> WasmLd<'a> {
    fn new(cmd: Command, sess: &'a Session) -> WasmLd<'a> {
        // If the atomics feature is enabled for wasm then we need a whole bunch
        // of flags:
        //
        // * `--shared-memory` - the link won't even succeed without this, flags
        //   the one linear memory as `shared`
        //
        // * `--max-memory=1G` - when specifying a shared memory this must also
        //   be specified. We conservatively choose 1GB but users should be able
        //   to override this with `-C link-arg`.
        //
        // * `--import-memory` - it doesn't make much sense for memory to be
        //   exported in a threaded module because typically you're
        //   sharing memory and instantiating the module multiple times. As a
        //   result if it were exported then we'd just have no sharing.
        //
        // On wasm32-unknown-unknown, we also export symbols for glue code to use:
        //    * `--export=*tls*` - when `#[thread_local]` symbols are used these
        //      symbols are how the TLS segments are initialized and configured.
        let mut wasm_ld = WasmLd { cmd, sess };
        if sess.target_features.contains(&sym::atomics) {
            wasm_ld.link_args(&["--shared-memory", "--max-memory=1073741824", "--import-memory"]);
            if sess.target.os == "unknown" || sess.target.os == "none" {
                wasm_ld.link_args(&[
                    "--export=__wasm_init_tls",
                    "--export=__tls_size",
                    "--export=__tls_align",
                    "--export=__tls_base",
                ]);
            }
        }
        wasm_ld
    }
}

impl<'a> Linker for WasmLd<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        output_kind: LinkOutputKind,
        _crate_type: CrateType,
        _out_filename: &Path,
    ) {
        match output_kind {
            LinkOutputKind::DynamicNoPicExe
            | LinkOutputKind::DynamicPicExe
            | LinkOutputKind::StaticNoPicExe
            | LinkOutputKind::StaticPicExe => {}
            LinkOutputKind::DynamicDylib | LinkOutputKind::StaticDylib => {
                self.link_arg("--no-entry");
            }
            LinkOutputKind::WasiReactorExe => {
                self.link_args(&["--entry", "_initialize"]);
            }
        }
    }

    fn link_dylib_by_name(&mut self, name: &str, _verbatim: bool, _as_needed: bool) {
        self.link_or_cc_args(&["-l", name]);
    }

    fn link_dylib_by_path(&mut self, path: &Path, _as_needed: bool) {
        self.link_or_cc_arg(path);
    }

    fn link_staticlib_by_name(&mut self, name: &str, _verbatim: bool, whole_archive: bool) {
        if !whole_archive {
            self.link_or_cc_args(&["-l", name]);
        } else {
            self.link_arg("--whole-archive")
                .link_or_cc_args(&["-l", name])
                .link_arg("--no-whole-archive");
        }
    }

    fn link_staticlib_by_path(&mut self, path: &Path, whole_archive: bool) {
        if !whole_archive {
            self.link_or_cc_arg(path);
        } else {
            self.link_arg("--whole-archive").link_or_cc_arg(path).link_arg("--no-whole-archive");
        }
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn gc_sections(&mut self, _keep_metadata: bool) {
        self.link_arg("--gc-sections");
    }

    fn no_gc_sections(&mut self) {
        self.link_arg("--no-gc-sections");
    }

    fn optimize(&mut self) {
        // The -O flag is, as of late 2023, only used for merging of strings and debuginfo, and
        // only differentiates -O0 and -O1. It does not apply to LTO.
        self.link_arg(match self.sess.opts.optimize {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::More => "-O2",
            OptLevel::Aggressive => "-O3",
            // Currently LLD doesn't support `Os` and `Oz`, so pass through `O2`
            // instead.
            OptLevel::Size => "-O2",
            OptLevel::SizeMin => "-O2",
        });
    }

    fn pgo_gen(&mut self) {}

    fn debuginfo(&mut self, strip: Strip, _: &[PathBuf]) {
        match strip {
            Strip::None => {}
            Strip::Debuginfo => {
                self.link_arg("--strip-debug");
            }
            Strip::Symbols => {
                self.link_arg("--strip-all");
            }
        }
    }

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn export_symbols(
        &mut self,
        _tmpdir: &Path,
        _crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        for (sym, _) in symbols {
            self.link_args(&["--export", sym]);
        }

        // LLD will hide these otherwise-internal symbols since it only exports
        // symbols explicitly passed via the `--export` flags above and hides all
        // others. Various bits and pieces of wasm32-unknown-unknown tooling use
        // this, so be sure these symbols make their way out of the linker as well.
        if self.sess.target.os == "unknown" || self.sess.target.os == "none" {
            self.link_args(&["--export=__heap_base", "--export=__data_end"]);
        }
    }

    fn subsystem(&mut self, _subsystem: &str) {}

    fn linker_plugin_lto(&mut self) {
        match self.sess.opts.cg.linker_plugin_lto {
            LinkerPluginLto::Disabled => {
                // Nothing to do
            }
            LinkerPluginLto::LinkerPluginAuto => {
                self.push_linker_plugin_lto_args();
            }
            LinkerPluginLto::LinkerPlugin(_) => {
                self.push_linker_plugin_lto_args();
            }
        }
    }
}

impl<'a> WasmLd<'a> {
    fn push_linker_plugin_lto_args(&mut self) {
        let opt_level = match self.sess.opts.optimize {
            config::OptLevel::No => "O0",
            config::OptLevel::Less => "O1",
            config::OptLevel::More => "O2",
            config::OptLevel::Aggressive => "O3",
            // wasm-ld only handles integer LTO opt levels. Use O2
            config::OptLevel::Size | config::OptLevel::SizeMin => "O2",
        };
        self.link_arg(&format!("--lto-{opt_level}"));
    }
}

/// Linker shepherd script for L4Re (Fiasco)
struct L4Bender<'a> {
    cmd: Command,
    sess: &'a Session,
    hinted_static: bool,
}

impl<'a> Linker for L4Bender<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        _output_kind: LinkOutputKind,
        _crate_type: CrateType,
        _out_filename: &Path,
    ) {
    }

    fn link_staticlib_by_name(&mut self, name: &str, _verbatim: bool, whole_archive: bool) {
        self.hint_static();
        if !whole_archive {
            self.link_arg(format!("-PC{name}"));
        } else {
            self.link_arg("--whole-archive")
                .link_or_cc_arg(format!("-l{name}"))
                .link_arg("--no-whole-archive");
        }
    }

    fn link_staticlib_by_path(&mut self, path: &Path, whole_archive: bool) {
        self.hint_static();
        if !whole_archive {
            self.link_or_cc_arg(path);
        } else {
            self.link_arg("--whole-archive").link_or_cc_arg(path).link_arg("--no-whole-archive");
        }
    }

    fn full_relro(&mut self) {
        self.link_args(&["-z", "relro", "-z", "now"]);
    }

    fn partial_relro(&mut self) {
        self.link_args(&["-z", "relro"]);
    }

    fn no_relro(&mut self) {
        self.link_args(&["-z", "norelro"]);
    }

    fn gc_sections(&mut self, keep_metadata: bool) {
        if !keep_metadata {
            self.link_arg("--gc-sections");
        }
    }

    fn no_gc_sections(&mut self) {
        self.link_arg("--no-gc-sections");
    }

    fn optimize(&mut self) {
        // GNU-style linkers support optimization with -O. GNU ld doesn't
        // need a numeric argument, but other linkers do.
        if self.sess.opts.optimize == config::OptLevel::More
            || self.sess.opts.optimize == config::OptLevel::Aggressive
        {
            self.link_arg("-O1");
        }
    }

    fn pgo_gen(&mut self) {}

    fn debuginfo(&mut self, strip: Strip, _: &[PathBuf]) {
        match strip {
            Strip::None => {}
            Strip::Debuginfo => {
                self.link_arg("--strip-debug");
            }
            Strip::Symbols => {
                self.link_arg("--strip-all");
            }
        }
    }

    fn no_default_libraries(&mut self) {
        self.cc_arg("-nostdlib");
    }

    fn export_symbols(&mut self, _: &Path, _: CrateType, _: &[(String, SymbolExportKind)]) {
        // ToDo, not implemented, copy from GCC
        self.sess.dcx().emit_warn(errors::L4BenderExportingSymbolsUnimplemented);
    }

    fn subsystem(&mut self, subsystem: &str) {
        self.link_arg(&format!("--subsystem {subsystem}"));
    }

    fn reset_per_library_state(&mut self) {
        self.hint_static(); // Reset to default before returning the composed command line.
    }

    fn linker_plugin_lto(&mut self) {}

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn no_crt_objects(&mut self) {}
}

impl<'a> L4Bender<'a> {
    fn new(cmd: Command, sess: &'a Session) -> L4Bender<'a> {
        L4Bender { cmd, sess, hinted_static: false }
    }

    fn hint_static(&mut self) {
        if !self.hinted_static {
            self.link_or_cc_arg("-static");
            self.hinted_static = true;
        }
    }
}

/// Linker for AIX.
struct AixLinker<'a> {
    cmd: Command,
    sess: &'a Session,
    hinted_static: Option<bool>,
}

impl<'a> AixLinker<'a> {
    fn new(cmd: Command, sess: &'a Session) -> AixLinker<'a> {
        AixLinker { cmd, sess, hinted_static: None }
    }

    fn hint_static(&mut self) {
        if self.hinted_static != Some(true) {
            self.link_arg("-bstatic");
            self.hinted_static = Some(true);
        }
    }

    fn hint_dynamic(&mut self) {
        if self.hinted_static != Some(false) {
            self.link_arg("-bdynamic");
            self.hinted_static = Some(false);
        }
    }

    fn build_dylib(&mut self, _out_filename: &Path) {
        self.link_args(&["-bM:SRE", "-bnoentry"]);
        // FIXME: Use CreateExportList utility to create export list
        // and remove -bexpfull.
        self.link_arg("-bexpfull");
    }
}

impl<'a> Linker for AixLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        output_kind: LinkOutputKind,
        _crate_type: CrateType,
        out_filename: &Path,
    ) {
        match output_kind {
            LinkOutputKind::DynamicDylib => {
                self.hint_dynamic();
                self.build_dylib(out_filename);
            }
            LinkOutputKind::StaticDylib => {
                self.hint_static();
                self.build_dylib(out_filename);
            }
            _ => {}
        }
    }

    fn link_dylib_by_name(&mut self, name: &str, verbatim: bool, _as_needed: bool) {
        self.hint_dynamic();
        self.link_or_cc_arg(if verbatim { String::from(name) } else { format!("-l{name}") });
    }

    fn link_dylib_by_path(&mut self, path: &Path, _as_needed: bool) {
        self.hint_dynamic();
        self.link_or_cc_arg(path);
    }

    fn link_staticlib_by_name(&mut self, name: &str, verbatim: bool, whole_archive: bool) {
        self.hint_static();
        if !whole_archive {
            self.link_or_cc_arg(if verbatim { String::from(name) } else { format!("-l{name}") });
        } else {
            let mut arg = OsString::from("-bkeepfile:");
            arg.push(find_native_static_library(name, verbatim, self.sess));
            self.link_or_cc_arg(arg);
        }
    }

    fn link_staticlib_by_path(&mut self, path: &Path, whole_archive: bool) {
        self.hint_static();
        if !whole_archive {
            self.link_or_cc_arg(path);
        } else {
            let mut arg = OsString::from("-bkeepfile:");
            arg.push(path);
            self.link_arg(arg);
        }
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn gc_sections(&mut self, _keep_metadata: bool) {
        self.link_arg("-bgc");
    }

    fn no_gc_sections(&mut self) {
        self.link_arg("-bnogc");
    }

    fn optimize(&mut self) {}

    fn pgo_gen(&mut self) {
        self.link_arg("-bdbg:namedsects:ss");
        self.link_arg("-u");
        self.link_arg("__llvm_profile_runtime");
    }

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn debuginfo(&mut self, _: Strip, _: &[PathBuf]) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn export_symbols(
        &mut self,
        tmpdir: &Path,
        _crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        let path = tmpdir.join("list.exp");
        let res: io::Result<()> = try {
            let mut f = File::create_buffered(&path)?;
            // FIXME: use llvm-nm to generate export list.
            for (symbol, _) in symbols {
                debug!("  _{symbol}");
                writeln!(f, "  {symbol}")?;
            }
        };
        if let Err(e) = res {
            self.sess.dcx().fatal(format!("failed to write export file: {e}"));
        }
        self.link_arg(format!("-bE:{}", path.to_str().unwrap()));
    }

    fn subsystem(&mut self, _subsystem: &str) {}

    fn reset_per_library_state(&mut self) {
        self.hint_dynamic();
    }

    fn linker_plugin_lto(&mut self) {}

    fn add_eh_frame_header(&mut self) {}

    fn add_no_exec(&mut self) {}

    fn add_as_needed(&mut self) {}
}

fn for_each_exported_symbols_include_dep<'tcx>(
    tcx: TyCtxt<'tcx>,
    crate_type: CrateType,
    mut callback: impl FnMut(ExportedSymbol<'tcx>, SymbolExportInfo, CrateNum),
) {
    let formats = tcx.dependency_formats(());
    let deps = &formats[&crate_type];

    for (cnum, dep_format) in deps.iter_enumerated() {
        // For each dependency that we are linking to statically ...
        if *dep_format == Linkage::Static {
            for &(symbol, info) in tcx.exported_non_generic_symbols(cnum).iter() {
                callback(symbol, info, cnum);
            }
            for &(symbol, info) in tcx.exported_generic_symbols(cnum).iter() {
                callback(symbol, info, cnum);
            }
        }
    }
}

pub(crate) fn exported_symbols(
    tcx: TyCtxt<'_>,
    crate_type: CrateType,
) -> Vec<(String, SymbolExportKind)> {
    if let Some(ref exports) = tcx.sess.target.override_export_symbols {
        return exports
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    // FIXME use the correct export kind for this symbol. override_export_symbols
                    // can't directly specify the SymbolExportKind as it is defined in rustc_middle
                    // which rustc_target can't depend on.
                    SymbolExportKind::Text,
                )
            })
            .collect();
    }

    if let CrateType::ProcMacro = crate_type {
        exported_symbols_for_proc_macro_crate(tcx)
    } else {
        exported_symbols_for_non_proc_macro(tcx, crate_type)
    }
}

fn exported_symbols_for_non_proc_macro(
    tcx: TyCtxt<'_>,
    crate_type: CrateType,
) -> Vec<(String, SymbolExportKind)> {
    let mut symbols = Vec::new();
    let export_threshold = symbol_export::crates_export_threshold(&[crate_type]);
    for_each_exported_symbols_include_dep(tcx, crate_type, |symbol, info, cnum| {
        // Do not export mangled symbols from cdylibs and don't attempt to export compiler-builtins
        // from any cdylib. The latter doesn't work anyway as we use hidden visibility for
        // compiler-builtins. Most linkers silently ignore it, but ld64 gives a warning.
        if info.level.is_below_threshold(export_threshold) && !tcx.is_compiler_builtins(cnum) {
            symbols.push((
                symbol_export::exporting_symbol_name_for_instance_in_crate(tcx, symbol, cnum),
                info.kind,
            ));
            symbol_export::extend_exported_symbols(&mut symbols, tcx, symbol, cnum);
        }
    });

    symbols
}

fn exported_symbols_for_proc_macro_crate(tcx: TyCtxt<'_>) -> Vec<(String, SymbolExportKind)> {
    // `exported_symbols` will be empty when !should_codegen.
    if !tcx.sess.opts.output_types.should_codegen() {
        return Vec::new();
    }

    let stable_crate_id = tcx.stable_crate_id(LOCAL_CRATE);
    let proc_macro_decls_name = tcx.sess.generate_proc_macro_decls_symbol(stable_crate_id);
    let metadata_symbol_name = exported_symbols::metadata_symbol_name(tcx);

    vec![
        (proc_macro_decls_name, SymbolExportKind::Data),
        (metadata_symbol_name, SymbolExportKind::Data),
    ]
}

pub(crate) fn linked_symbols(
    tcx: TyCtxt<'_>,
    crate_type: CrateType,
) -> Vec<(String, SymbolExportKind)> {
    match crate_type {
        CrateType::Executable
        | CrateType::ProcMacro
        | CrateType::Cdylib
        | CrateType::Dylib
        | CrateType::Sdylib => (),
        CrateType::Staticlib | CrateType::Rlib => {
            // These are not linked, so no need to generate symbols.o for them.
            return Vec::new();
        }
    }

    match tcx.sess.lto() {
        Lto::No | Lto::ThinLocal => {}
        Lto::Thin | Lto::Fat => {
            // We really only need symbols from upstream rlibs to end up in the linked symbols list.
            // The rest are in separate object files which the linker will always link in and
            // doesn't have rules around the order in which they need to appear.
            // When doing LTO, some of the symbols in the linked symbols list happen to be
            // internalized by LTO, which then prevents referencing them from symbols.o. When doing
            // LTO, all object files that get linked in will be local object files rather than
            // pulled in from rlibs, so an empty linked symbols list works fine to avoid referencing
            // all those internalized symbols from symbols.o.
            return Vec::new();
        }
    }

    let mut symbols = Vec::new();

    let export_threshold = symbol_export::crates_export_threshold(&[crate_type]);
    for_each_exported_symbols_include_dep(tcx, crate_type, |symbol, info, cnum| {
        if info.level.is_below_threshold(export_threshold) && !tcx.is_compiler_builtins(cnum)
            || info.used
            || info.rustc_std_internal_symbol
        {
            symbols.push((
                symbol_export::linking_symbol_name_for_instance_in_crate(
                    tcx, symbol, info.kind, cnum,
                ),
                info.kind,
            ));
        }
    });

    symbols
}

/// Much simplified and explicit CLI for the NVPTX linker. The linker operates
/// with bitcode and uses LLVM backend to generate a PTX assembly.
struct PtxLinker<'a> {
    cmd: Command,
    sess: &'a Session,
}

impl<'a> Linker for PtxLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        _output_kind: LinkOutputKind,
        _crate_type: CrateType,
        _out_filename: &Path,
    ) {
    }

    fn link_staticlib_by_name(&mut self, _name: &str, _verbatim: bool, _whole_archive: bool) {
        panic!("staticlibs not supported")
    }

    fn link_staticlib_by_path(&mut self, path: &Path, _whole_archive: bool) {
        self.link_arg("--rlib").link_arg(path);
    }

    fn debuginfo(&mut self, _strip: Strip, _: &[PathBuf]) {
        self.link_arg("--debug");
    }

    fn add_object(&mut self, path: &Path) {
        self.link_arg("--bitcode").link_arg(path);
    }

    fn optimize(&mut self) {
        match self.sess.lto() {
            Lto::Thin | Lto::Fat | Lto::ThinLocal => {
                self.link_arg("-Olto");
            }

            Lto::No => {}
        }
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn gc_sections(&mut self, _keep_metadata: bool) {}

    fn no_gc_sections(&mut self) {}

    fn pgo_gen(&mut self) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn export_symbols(
        &mut self,
        _tmpdir: &Path,
        _crate_type: CrateType,
        _symbols: &[(String, SymbolExportKind)],
    ) {
    }

    fn subsystem(&mut self, _subsystem: &str) {}

    fn linker_plugin_lto(&mut self) {}
}

/// The `self-contained` LLVM bitcode linker
struct LlbcLinker<'a> {
    cmd: Command,
    sess: &'a Session,
}

impl<'a> Linker for LlbcLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        _output_kind: LinkOutputKind,
        _crate_type: CrateType,
        _out_filename: &Path,
    ) {
    }

    fn link_staticlib_by_name(&mut self, _name: &str, _verbatim: bool, _whole_archive: bool) {
        panic!("staticlibs not supported")
    }

    fn link_staticlib_by_path(&mut self, path: &Path, _whole_archive: bool) {
        self.link_or_cc_arg(path);
    }

    fn debuginfo(&mut self, _strip: Strip, _: &[PathBuf]) {
        self.link_arg("--debug");
    }

    fn optimize(&mut self) {
        self.link_arg(match self.sess.opts.optimize {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::More => "-O2",
            OptLevel::Aggressive => "-O3",
            OptLevel::Size => "-Os",
            OptLevel::SizeMin => "-Oz",
        });
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn gc_sections(&mut self, _keep_metadata: bool) {}

    fn no_gc_sections(&mut self) {}

    fn pgo_gen(&mut self) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn export_symbols(
        &mut self,
        _tmpdir: &Path,
        _crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        match _crate_type {
            CrateType::Cdylib => {
                for (sym, _) in symbols {
                    self.link_args(&["--export-symbol", sym]);
                }
            }
            _ => (),
        }
    }

    fn subsystem(&mut self, _subsystem: &str) {}

    fn linker_plugin_lto(&mut self) {}
}

struct BpfLinker<'a> {
    cmd: Command,
    sess: &'a Session,
}

impl<'a> Linker for BpfLinker<'a> {
    fn cmd(&mut self) -> &mut Command {
        &mut self.cmd
    }

    fn set_output_kind(
        &mut self,
        _output_kind: LinkOutputKind,
        _crate_type: CrateType,
        _out_filename: &Path,
    ) {
    }

    fn link_staticlib_by_name(&mut self, _name: &str, _verbatim: bool, _whole_archive: bool) {
        panic!("staticlibs not supported")
    }

    fn link_staticlib_by_path(&mut self, path: &Path, _whole_archive: bool) {
        self.link_or_cc_arg(path);
    }

    fn debuginfo(&mut self, _strip: Strip, _: &[PathBuf]) {
        self.link_arg("--debug");
    }

    fn optimize(&mut self) {
        self.link_arg(match self.sess.opts.optimize {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::More => "-O2",
            OptLevel::Aggressive => "-O3",
            OptLevel::Size => "-Os",
            OptLevel::SizeMin => "-Oz",
        });
    }

    fn full_relro(&mut self) {}

    fn partial_relro(&mut self) {}

    fn no_relro(&mut self) {}

    fn gc_sections(&mut self, _keep_metadata: bool) {}

    fn no_gc_sections(&mut self) {}

    fn pgo_gen(&mut self) {}

    fn no_crt_objects(&mut self) {}

    fn no_default_libraries(&mut self) {}

    fn control_flow_guard(&mut self) {}

    fn ehcont_guard(&mut self) {}

    fn export_symbols(
        &mut self,
        tmpdir: &Path,
        _crate_type: CrateType,
        symbols: &[(String, SymbolExportKind)],
    ) {
        let path = tmpdir.join("symbols");
        let res: io::Result<()> = try {
            let mut f = File::create_buffered(&path)?;
            for (sym, _) in symbols {
                writeln!(f, "{sym}")?;
            }
        };
        if let Err(error) = res {
            self.sess.dcx().emit_fatal(errors::SymbolFileWriteFailure { error });
        } else {
            self.link_arg("--export-symbols").link_arg(&path);
        }
    }

    fn subsystem(&mut self, _subsystem: &str) {}

    fn linker_plugin_lto(&mut self) {}
}
