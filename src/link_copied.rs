//! All functions here are copied from https://github.com/rust-lang/rust/blob/942864a000efd74b73e36bda5606b2cdb55ecf39/src/librustc_codegen_llvm/back/link.rs

use std::fmt;
use std::fs;
use std::io;
use std::iter;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};

use log::info;

use rustc::middle::cstore::{NativeLibrary, NativeLibraryKind};
use rustc::middle::dependency_format::Linkage;
use rustc::session::config::{self, OutputType, RUST_CGU_EXT};
use rustc::session::search_paths::PathKind;
use rustc::session::Session;
use rustc::util::common::time;
use rustc_codegen_ssa::back::command::Command;
use rustc_codegen_ssa::back::linker::*;
use rustc_codegen_ssa::back::link::*;
use rustc_data_structures::fx::FxHashSet;
use rustc_fs_util::fix_windows_verbatim_for_gcc;
use syntax::attr;

use crate::prelude::*;

use crate::archive::{ArchiveBuilder, ArchiveConfig};
use crate::metadata::METADATA_FILENAME;


// cg_clif doesn't have bytecode, so this is just a dummy
const RLIB_BYTECODE_EXTENSION: &str = ".cg_clif_bytecode_dummy";

fn archive_search_paths(sess: &Session) -> Vec<PathBuf> {
    let mut search = Vec::new();
    sess.target_filesearch(PathKind::Native).for_each_lib_search_path(|path, _| {
        search.push(path.to_path_buf());
    });
    return search;
}

fn archive_config<'a>(sess: &'a Session,
                      output: &Path,
                      input: Option<&Path>) -> ArchiveConfig<'a> {
    ArchiveConfig {
        sess,
        dst: output.to_path_buf(),
        src: input.map(|p| p.to_path_buf()),
        lib_search_paths: archive_search_paths(sess),
    }
}

pub fn exec_linker(sess: &Session, cmd: &mut Command, out_filename: &Path, tmpdir: &Path)
    -> io::Result<Output>
{
    // When attempting to spawn the linker we run a risk of blowing out the
    // size limits for spawning a new process with respect to the arguments
    // we pass on the command line.
    //
    // Here we attempt to handle errors from the OS saying "your list of
    // arguments is too big" by reinvoking the linker again with an `@`-file
    // that contains all the arguments. The theory is that this is then
    // accepted on all linkers and the linker will read all its options out of
    // there instead of looking at the command line.
    if !cmd.very_likely_to_exceed_some_spawn_limit() {
        match cmd.command().stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(child) => {
                let output = child.wait_with_output();
                flush_linked_file(&output, out_filename)?;
                return output;
            }
            Err(ref e) if command_line_too_big(e) => {
                info!("command line to linker was too big: {}", e);
            }
            Err(e) => return Err(e)
        }
    }

    info!("falling back to passing arguments to linker via an @-file");
    let mut cmd2 = cmd.clone();
    let mut args = String::new();
    for arg in cmd2.take_args() {
        args.push_str(&Escape {
            arg: arg.to_str().unwrap(),
            is_like_msvc: sess.target.target.options.is_like_msvc,
        }.to_string());
        args.push_str("\n");
    }
    let file = tmpdir.join("linker-arguments");
    let bytes = if sess.target.target.options.is_like_msvc {
        let mut out = Vec::with_capacity((1 + args.len()) * 2);
        // start the stream with a UTF-16 BOM
        for c in iter::once(0xFEFF).chain(args.encode_utf16()) {
            // encode in little endian
            out.push(c as u8);
            out.push((c >> 8) as u8);
        }
        out
    } else {
        args.into_bytes()
    };
    fs::write(&file, &bytes)?;
    cmd2.arg(format!("@{}", file.display()));
    info!("invoking linker {:?}", cmd2);
    let output = cmd2.output();
    flush_linked_file(&output, out_filename)?;
    return output;

    #[cfg(unix)]
    fn flush_linked_file(_: &io::Result<Output>, _: &Path) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn flush_linked_file(command_output: &io::Result<Output>, out_filename: &Path)
        -> io::Result<()>
    {
        // On Windows, under high I/O load, output buffers are sometimes not flushed,
        // even long after process exit, causing nasty, non-reproducible output bugs.
        //
        // File::sync_all() calls FlushFileBuffers() down the line, which solves the problem.
        //
        // А full writeup of the original Chrome bug can be found at
        // randomascii.wordpress.com/2018/02/25/compiler-bug-linker-bug-windows-kernel-bug/amp

        if let &Ok(ref out) = command_output {
            if out.status.success() {
                if let Ok(of) = fs::OpenOptions::new().write(true).open(out_filename) {
                    of.sync_all()?;
                }
            }
        }

        Ok(())
    }

    #[cfg(unix)]
    fn command_line_too_big(err: &io::Error) -> bool {
        err.raw_os_error() == Some(::libc::E2BIG)
    }

    #[cfg(windows)]
    fn command_line_too_big(err: &io::Error) -> bool {
        const ERROR_FILENAME_EXCED_RANGE: i32 = 206;
        err.raw_os_error() == Some(ERROR_FILENAME_EXCED_RANGE)
    }

    struct Escape<'a> {
        arg: &'a str,
        is_like_msvc: bool,
    }

    impl<'a> fmt::Display for Escape<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.is_like_msvc {
                // This is "documented" at
                // https://msdn.microsoft.com/en-us/library/4xdcbak7.aspx
                //
                // Unfortunately there's not a great specification of the
                // syntax I could find online (at least) but some local
                // testing showed that this seemed sufficient-ish to catch
                // at least a few edge cases.
                write!(f, "\"")?;
                for c in self.arg.chars() {
                    match c {
                        '"' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
                write!(f, "\"")?;
            } else {
                // This is documented at https://linux.die.net/man/1/ld, namely:
                //
                // > Options in file are separated by whitespace. A whitespace
                // > character may be included in an option by surrounding the
                // > entire option in either single or double quotes. Any
                // > character (including a backslash) may be included by
                // > prefixing the character to be included with a backslash.
                //
                // We put an argument on each line, so all we need to do is
                // ensure the line is interpreted as one whole argument.
                for c in self.arg.chars() {
                    match c {
                        '\\' |
                        ' ' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
            }
            Ok(())
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
pub fn add_upstream_rust_crates(cmd: &mut dyn Linker,
                            sess: &Session,
                            codegen_results: &CodegenResults,
                            crate_type: config::CrateType,
                            tmpdir: &Path) {
    // All of the heavy lifting has previously been accomplished by the
    // dependency_format module of the compiler. This is just crawling the
    // output of that module, adding crates as necessary.
    //
    // Linking to a rlib involves just passing it to the linker (the linker
    // will slurp up the object files inside), and linking to a dynamic library
    // involves just passing the right -l flag.

    let formats = sess.dependency_formats.borrow();
    let data = formats.get(&crate_type).unwrap();

    // Invoke get_used_crates to ensure that we get a topological sorting of
    // crates.
    let deps = &codegen_results.crate_info.used_crates_dynamic;

    // There's a few internal crates in the standard library (aka libcore and
    // libstd) which actually have a circular dependence upon one another. This
    // currently arises through "weak lang items" where libcore requires things
    // like `rust_begin_unwind` but libstd ends up defining it. To get this
    // circular dependence to work correctly in all situations we'll need to be
    // sure to correctly apply the `--start-group` and `--end-group` options to
    // GNU linkers, otherwise if we don't use any other symbol from the standard
    // library it'll get discarded and the whole application won't link.
    //
    // In this loop we're calculating the `group_end`, after which crate to
    // pass `--end-group` and `group_start`, before which crate to pass
    // `--start-group`. We currently do this by passing `--end-group` after
    // the first crate (when iterating backwards) that requires a lang item
    // defined somewhere else. Once that's set then when we've defined all the
    // necessary lang items we'll pass `--start-group`.
    //
    // Note that this isn't amazing logic for now but it should do the trick
    // for the current implementation of the standard library.
    let mut group_end = None;
    let mut group_start = None;
    let mut end_with = FxHashSet::default();
    let info = &codegen_results.crate_info;
    for &(cnum, _) in deps.iter().rev() {
        if let Some(missing) = info.missing_lang_items.get(&cnum) {
            end_with.extend(missing.iter().cloned());
            if end_with.len() > 0 && group_end.is_none() {
                group_end = Some(cnum);
            }
        }
        end_with.retain(|item| info.lang_item_to_crate.get(item) != Some(&cnum));
        if end_with.len() == 0 && group_end.is_some() {
            group_start = Some(cnum);
            break
        }
    }

    // If we didn't end up filling in all lang items from upstream crates then
    // we'll be filling it in with our crate. This probably means we're the
    // standard library itself, so skip this for now.
    if group_end.is_some() && group_start.is_none() {
        group_end = None;
    }

    let mut compiler_builtins = None;

    for &(cnum, _) in deps.iter() {
        if group_start == Some(cnum) {
            cmd.group_start();
        }

        // We may not pass all crates through to the linker. Some crates may
        // appear statically in an existing dylib, meaning we'll pick up all the
        // symbols from the dylib.
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        match data[cnum.as_usize() - 1] {
            _ if codegen_results.crate_info.profiler_runtime == Some(cnum) => {
                add_static_crate(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
            }
            _ if codegen_results.crate_info.sanitizer_runtime == Some(cnum) => {
                link_sanitizer_runtime(cmd, sess, codegen_results, tmpdir, cnum);
            }
            // compiler-builtins are always placed last to ensure that they're
            // linked correctly.
            _ if codegen_results.crate_info.compiler_builtins == Some(cnum) => {
                assert!(compiler_builtins.is_none());
                compiler_builtins = Some(cnum);
            }
            Linkage::NotLinked |
            Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                add_static_crate(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
            }
            Linkage::Dynamic => {
                add_dynamic_crate(cmd, sess, &src.dylib.as_ref().unwrap().0)
            }
        }

        if group_end == Some(cnum) {
            cmd.group_end();
        }
    }

    // compiler-builtins are always placed last to ensure that they're
    // linked correctly.
    // We must always link the `compiler_builtins` crate statically. Even if it
    // was already "included" in a dylib (e.g. `libstd` when `-C prefer-dynamic`
    // is used)
    if let Some(cnum) = compiler_builtins {
        add_static_crate(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
    }

    // Converts a library file-stem into a cc -l argument
    fn unlib<'a>(config: &config::Config, stem: &'a str) -> &'a str {
        if stem.starts_with("lib") && !config.target.options.is_like_windows {
            &stem[3..]
        } else {
            stem
        }
    }

    // We must link the sanitizer runtime using -Wl,--whole-archive but since
    // it's packed in a .rlib, it contains stuff that are not objects that will
    // make the linker error. So we must remove those bits from the .rlib before
    // linking it.
    fn link_sanitizer_runtime(cmd: &mut dyn Linker,
                              sess: &Session,
                              codegen_results: &CodegenResults,
                              tmpdir: &Path,
                              cnum: CrateNum) {
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        let cratepath = &src.rlib.as_ref().unwrap().0;

        if sess.target.target.options.is_like_osx {
            // On Apple platforms, the sanitizer is always built as a dylib, and
            // LLVM will link to `@rpath/*.dylib`, so we need to specify an
            // rpath to the library as well (the rpath should be absolute, see
            // PR #41352 for details).
            //
            // FIXME: Remove this logic into librustc_*san once Cargo supports it
            let rpath = cratepath.parent().unwrap();
            let rpath = rpath.to_str().expect("non-utf8 component in path");
            cmd.args(&["-Wl,-rpath".into(), "-Xlinker".into(), rpath.into()]);
        }

        let dst = tmpdir.join(cratepath.file_name().unwrap());
        let cfg = archive_config(sess, &dst, Some(cratepath));
        let mut archive = ArchiveBuilder::new(cfg);
        archive.update_symbols();

        for f in archive.src_files() {
            if f.ends_with(RLIB_BYTECODE_EXTENSION) || f == METADATA_FILENAME {
                archive.remove_file(&f);
                continue
            }
        }

        archive.build();

        cmd.link_whole_rlib(&dst);
    }

    // Adds the static "rlib" versions of all crates to the command line.
    // There's a bit of magic which happens here specifically related to LTO and
    // dynamic libraries. Specifically:
    //
    // * For LTO, we remove upstream object files.
    // * For dylibs we remove metadata and bytecode from upstream rlibs
    //
    // When performing LTO, almost(*) all of the bytecode from the upstream
    // libraries has already been included in our object file output. As a
    // result we need to remove the object files in the upstream libraries so
    // the linker doesn't try to include them twice (or whine about duplicate
    // symbols). We must continue to include the rest of the rlib, however, as
    // it may contain static native libraries which must be linked in.
    //
    // (*) Crates marked with `#![no_builtins]` don't participate in LTO and
    // their bytecode wasn't included. The object files in those libraries must
    // still be passed to the linker.
    //
    // When making a dynamic library, linkers by default don't include any
    // object files in an archive if they're not necessary to resolve the link.
    // We basically want to convert the archive (rlib) to a dylib, though, so we
    // *do* want everything included in the output, regardless of whether the
    // linker thinks it's needed or not. As a result we must use the
    // --whole-archive option (or the platform equivalent). When using this
    // option the linker will fail if there are non-objects in the archive (such
    // as our own metadata and/or bytecode). All in all, for rlibs to be
    // entirely included in dylibs, we need to remove all non-object files.
    //
    // Note, however, that if we're not doing LTO or we're not producing a dylib
    // (aka we're making an executable), we can just pass the rlib blindly to
    // the linker (fast) because it's fine if it's not actually included as
    // we're at the end of the dependency chain.
    fn add_static_crate(cmd: &mut dyn Linker,
                        sess: &Session,
                        codegen_results: &CodegenResults,
                        tmpdir: &Path,
                        crate_type: config::CrateType,
                        cnum: CrateNum) {
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        let cratepath = &src.rlib.as_ref().unwrap().0;

        // See the comment above in `link_staticlib` and `link_rlib` for why if
        // there's a static library that's not relevant we skip all object
        // files.
        let native_libs = &codegen_results.crate_info.native_libraries[&cnum];
        let skip_native = native_libs.iter().any(|lib| {
            lib.kind == NativeLibraryKind::NativeStatic && !relevant_lib(sess, lib)
        });

        if (!are_upstream_rust_objects_already_included(sess) ||
            ignored_for_lto(sess, &codegen_results.crate_info, cnum)) &&
           crate_type != config::CrateType::Dylib &&
           !skip_native {
            cmd.link_rlib(&fix_windows_verbatim_for_gcc(cratepath));
            return
        }

        let dst = tmpdir.join(cratepath.file_name().unwrap());
        let name = cratepath.file_name().unwrap().to_str().unwrap();
        let name = &name[3..name.len() - 5]; // chop off lib/.rlib

        time(sess, &format!("altering {}.rlib", name), || {
            let cfg = archive_config(sess, &dst, Some(cratepath));
            let mut archive = ArchiveBuilder::new(cfg);
            archive.update_symbols();

            let mut any_objects = false;
            for f in archive.src_files() {
                if f.ends_with(RLIB_BYTECODE_EXTENSION) || f == METADATA_FILENAME {
                    archive.remove_file(&f);
                    continue
                }

                let canonical = f.replace("-", "_");
                let canonical_name = name.replace("-", "_");

                // Look for `.rcgu.o` at the end of the filename to conclude
                // that this is a Rust-related object file.
                fn looks_like_rust(s: &str) -> bool {
                    let path = Path::new(s);
                    let ext = path.extension().and_then(|s| s.to_str());
                    if ext != Some(OutputType::Object.extension()) {
                        return false
                    }
                    let ext2 = path.file_stem()
                        .and_then(|s| Path::new(s).extension())
                        .and_then(|s| s.to_str());
                    ext2 == Some(RUST_CGU_EXT)
                }

                let is_rust_object =
                    canonical.starts_with(&canonical_name) &&
                    looks_like_rust(&f);

                // If we've been requested to skip all native object files
                // (those not generated by the rust compiler) then we can skip
                // this file. See above for why we may want to do this.
                let skip_because_cfg_say_so = skip_native && !is_rust_object;

                // If we're performing LTO and this is a rust-generated object
                // file, then we don't need the object file as it's part of the
                // LTO module. Note that `#![no_builtins]` is excluded from LTO,
                // though, so we let that object file slide.
                let skip_because_lto = are_upstream_rust_objects_already_included(sess) &&
                    is_rust_object &&
                    (sess.target.target.options.no_builtins ||
                     !codegen_results.crate_info.is_no_builtins.contains(&cnum));

                if skip_because_cfg_say_so || skip_because_lto {
                    archive.remove_file(&f);
                } else {
                    any_objects = true;
                }
            }

            if !any_objects {
                return
            }
            archive.build();

            // If we're creating a dylib, then we need to include the
            // whole of each object in our archive into that artifact. This is
            // because a `dylib` can be reused as an intermediate artifact.
            //
            // Note, though, that we don't want to include the whole of a
            // compiler-builtins crate (e.g. compiler-rt) because it'll get
            // repeatedly linked anyway.
            if crate_type == config::CrateType::Dylib &&
                codegen_results.crate_info.compiler_builtins != Some(cnum) {
                cmd.link_whole_rlib(&fix_windows_verbatim_for_gcc(&dst));
            } else {
                cmd.link_rlib(&fix_windows_verbatim_for_gcc(&dst));
            }
        });
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crate(cmd: &mut dyn Linker, sess: &Session, cratepath: &Path) {
        // If we're performing LTO, then it should have been previously required
        // that all upstream rust dependencies were available in an rlib format.
        assert!(!are_upstream_rust_objects_already_included(sess));

        // Just need to tell the linker about where the library lives and
        // what its name is
        let parent = cratepath.parent();
        if let Some(dir) = parent {
            cmd.include_path(&fix_windows_verbatim_for_gcc(dir));
        }
        let filestem = cratepath.file_stem().unwrap().to_str().unwrap();
        cmd.link_rust_dylib(&unlib(&sess.target, filestem),
                            parent.unwrap_or(Path::new("")));
    }
}

// # Native library linking
//
// User-supplied library search paths (-L on the command line). These are
// the same paths used to find Rust crates, so some of them may have been
// added already by the previous crate linking code. This only allows them
// to be found at compile time so it is still entirely up to outside
// forces to make sure that library can be found at runtime.
//
// Also note that the native libraries linked here are only the ones located
// in the current crate. Upstream crates with native library dependencies
// may have their native library pulled in above.
pub fn add_local_native_libraries(cmd: &mut dyn Linker,
                              sess: &Session,
                              codegen_results: &CodegenResults) {
    sess.target_filesearch(PathKind::All).for_each_lib_search_path(|path, k| {
        match k {
            PathKind::Framework => { cmd.framework_path(path); }
            _ => { cmd.include_path(&fix_windows_verbatim_for_gcc(path)); }
        }
    });

    let relevant_libs = codegen_results.crate_info.used_libraries.iter().filter(|l| {
        relevant_lib(sess, l)
    });

    let search_path = archive_search_paths(sess);
    for lib in relevant_libs {
        let name = match lib.name {
            Some(ref l) => l,
            None => continue,
        };
        match lib.kind {
            NativeLibraryKind::NativeUnknown => cmd.link_dylib(&name.as_str()),
            NativeLibraryKind::NativeFramework => cmd.link_framework(&name.as_str()),
            NativeLibraryKind::NativeStaticNobundle => cmd.link_staticlib(&name.as_str()),
            NativeLibraryKind::NativeStatic => cmd.link_whole_staticlib(&name.as_str(),
                                                                        &search_path)
        }
    }
}

// Link in all of our upstream crates' native dependencies. Remember that
// all of these upstream native dependencies are all non-static
// dependencies. We've got two cases then:
//
// 1. The upstream crate is an rlib. In this case we *must* link in the
// native dependency because the rlib is just an archive.
//
// 2. The upstream crate is a dylib. In order to use the dylib, we have to
// have the dependency present on the system somewhere. Thus, we don't
// gain a whole lot from not linking in the dynamic dependency to this
// crate as well.
//
// The use case for this is a little subtle. In theory the native
// dependencies of a crate are purely an implementation detail of the crate
// itself, but the problem arises with generic and inlined functions. If a
// generic function calls a native function, then the generic function must
// be instantiated in the target crate, meaning that the native symbol must
// also be resolved in the target crate.
pub fn add_upstream_native_libraries(cmd: &mut dyn Linker,
                                 sess: &Session,
                                 codegen_results: &CodegenResults,
                                 crate_type: config::CrateType) {
    // Be sure to use a topological sorting of crates because there may be
    // interdependencies between native libraries. When passing -nodefaultlibs,
    // for example, almost all native libraries depend on libc, so we have to
    // make sure that's all the way at the right (liblibc is near the base of
    // the dependency chain).
    //
    // This passes RequireStatic, but the actual requirement doesn't matter,
    // we're just getting an ordering of crate numbers, we're not worried about
    // the paths.
    let formats = sess.dependency_formats.borrow();
    let data = formats.get(&crate_type).unwrap();

    let crates = &codegen_results.crate_info.used_crates_static;
    for &(cnum, _) in crates {
        for lib in codegen_results.crate_info.native_libraries[&cnum].iter() {
            let name = match lib.name {
                Some(ref l) => l,
                None => continue,
            };
            if !relevant_lib(sess, &lib) {
                continue
            }
            match lib.kind {
                NativeLibraryKind::NativeUnknown => cmd.link_dylib(&name.as_str()),
                NativeLibraryKind::NativeFramework => cmd.link_framework(&name.as_str()),
                NativeLibraryKind::NativeStaticNobundle => {
                    // Link "static-nobundle" native libs only if the crate they originate from
                    // is being linked statically to the current crate.  If it's linked dynamically
                    // or is an rlib already included via some other dylib crate, the symbols from
                    // native libs will have already been included in that dylib.
                    if data[cnum.as_usize() - 1] == Linkage::Static {
                        cmd.link_staticlib(&name.as_str())
                    }
                },
                // ignore statically included native libraries here as we've
                // already included them when we included the rust library
                // previously
                NativeLibraryKind::NativeStatic => {}
            }
        }
    }
}

fn relevant_lib(sess: &Session, lib: &NativeLibrary) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

fn are_upstream_rust_objects_already_included(sess: &Session) -> bool {
    match sess.lto() {
        Lto::Fat => true,
        Lto::Thin => {
            // If we defer LTO to the linker, we haven't run LTO ourselves, so
            // any upstream object files have not been copied yet.
            !sess.opts.debugging_opts.cross_lang_lto.enabled()
        }
        Lto::No |
        Lto::ThinLocal => false,
    }
}
