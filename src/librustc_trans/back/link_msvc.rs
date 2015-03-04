// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::archive::{Archive, ArchiveConfig};
use super::archive;
use super::rpath;
use super::rpath::RPathConfig;

use session::config;
use session::config::NoDebugInfo;
use session::search_paths::PathKind;
use session::Session;
use metadata::{cstore, filesearch, csearch};
use metadata::filesearch::FileDoesntMatch;
use trans::CrateTranslation;
use util::common::time;

use std::str;
use std::old_io::{fs, TempDir, Command};
use std::old_io;

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
pub fn link_natively(sess: &Session, trans: &CrateTranslation, dylib: bool,
                 obj_filename: &Path, out_filename: &Path) {
    let tmpdir = TempDir::new("rustc").ok().expect("needs a temp dir");

    // The invocations of cc share some flags across platforms
    let pname = super::link::get_cc_prog(sess);
    let mut cmd = Command::new(&pname[..]);

    cmd.args(&sess.target.target.options.pre_link_args[]);
    link_args(&mut cmd, sess, dylib, tmpdir.path(),
              trans, obj_filename, out_filename);
    cmd.args(&sess.target.target.options.post_link_args[]);
    if !sess.target.target.options.no_compiler_rt {
        cmd.arg("msvcrt.lib");
        cmd.arg("compiler-rt.lib");
    }

    if sess.opts.debugging_opts.print_link_args {
        println!("{:?}", &cmd);
    }

    // May have not found libraries in the right formats.
    sess.abort_if_errors();

    // Invoke the system linker
    debug!("{:?}", &cmd);
    let prog = time(sess.time_passes(), "running linker", (), |()| cmd.output());
    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                sess.err(&format!("linking with `{}` failed: {}",
                                 pname,
                                 prog.status)[]);
                sess.note(&format!("{:?}", &cmd)[]);
                let mut output = prog.error.clone();
                output.push_all(&prog.output[]);
                sess.note(str::from_utf8(&output[..]).unwrap());
                sess.abort_if_errors();
            }
            debug!("linker stderr:\n{}", String::from_utf8(prog.error).unwrap());
            debug!("linker stdout:\n{}", String::from_utf8(prog.output).unwrap());
        },
        Err(e) => {
            sess.err(&format!("could not exec the linker `{}`: {}",
                             pname,
                             e)[]);
            sess.abort_if_errors();
        }
    }


    // On OSX, debuggers need this utility to get run to do some munging of
    // the symbols
    if sess.target.target.options.is_like_osx && sess.opts.debuginfo != NoDebugInfo {
        match Command::new("dsymutil").arg(out_filename).output() {
            Ok(..) => {}
            Err(e) => {
                sess.err(&format!("failed to run dsymutil: {}", e)[]);
                sess.abort_if_errors();
            }
        }
    }
}

fn link_args(cmd: &mut Command,
             sess: &Session,
             dylib: bool,
             tmpdir: &Path,
             trans: &CrateTranslation,
             obj_filename: &Path,
             out_filename: &Path) {

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = sess.target_filesearch(PathKind::All).get_lib_path();

    // target descriptor
    let t = &sess.target.target;

    lib_path.as_str().map(|lp| cmd.arg(format!("/LIBPATH:{}", lp)));
    out_filename.as_str().map(|out| cmd.arg(format!("/OUT:{}", out)));
    
    cmd.arg(obj_filename);

    // Stack growth requires statically linking a __morestack function. Note
    // that this is listed *before* all other libraries. Due to the usage of the
    // --as-needed flag below, the standard library may only be useful for its
    // rust_stack_exhausted function. In this case, we must ensure that the
    // libmorestack.a file appears *before* the standard library (so we put it
    // at the very front).
    //
    // Most of the time this is sufficient, except for when LLVM gets super
    // clever. If, for example, we have a main function `fn main() {}`, LLVM
    // will optimize out calls to `__morestack` entirely because the function
    // doesn't need any stack at all!
    //
    // To get around this snag, we specially tell the linker to always include
    // all contents of this library. This way we're guaranteed that the linker
    // will include the __morestack symbol 100% of the time, always resolving
    // references to it even if the object above didn't use it.
    if t.options.morestack {
        cmd.arg("morestack.lib");
    }

    // When linking a dynamic library, we put the metadata into a section of the
    // executable. This metadata is in a separate object file from the main
    // object file, so we link that in here.
    if dylib {
        cmd.arg(obj_filename.with_extension("metadata.o"));
    }

    let used_link_args = sess.cstore.get_used_link_args().borrow();

    // We want to prevent the compiler from accidentally leaking in any system
    // libraries, so we explicitly ask gcc to not link to any libraries by
    // default. Note that this does not happen for windows because windows pulls
    // in some large number of libraries and I couldn't quite figure out which
    // subset we wanted.

    // We have to keep this in for now - since we need to link to the MSVCRT for
    // things such as jemalloc.
    //cmd.arg("/nodefaultlib");

    // Take careful note of the ordering of the arguments we pass to the linker
    // here. Linkers will assume that things on the left depend on things to the
    // right. Things on the right cannot depend on things on the left. This is
    // all formally implemented in terms of resolving symbols (libs on the right
    // resolve unknown symbols of libs on the left, but not vice versa).
    //
    // For this reason, we have organized the arguments we pass to the linker as
    // such:
    //
    //  1. The local object that LLVM just generated
    //  2. Upstream rust libraries
    //  3. Local native libraries
    //  4. Upstream native libraries
    //
    // This is generally fairly natural, but some may expect 2 and 3 to be
    // swapped. The reason that all native libraries are put last is that it's
    // not recommended for a native library to depend on a symbol from a rust
    // crate. If this is the case then a staticlib crate is recommended, solving
    // the problem.
    //
    // Additionally, it is occasionally the case that upstream rust libraries
    // depend on a local native library. In the case of libraries such as
    // lua/glfw/etc the name of the library isn't the same across all platforms,
    // so only the consumer crate of a library knows the actual name. This means
    // that downstream crates will provide the #[link] attribute which upstream
    // crates will depend on. Hence local native libraries are after out
    // upstream rust crates.
    //
    // In theory this means that a symbol in an upstream native library will be
    // shadowed by a local native library when it wouldn't have been before, but
    // this kind of behavior is pretty platform specific and generally not
    // recommended anyway, so I don't think we're shooting ourself in the foot
    // much with that.
    add_upstream_rust_crates(cmd, sess, dylib, tmpdir, trans);
    add_local_native_libraries(cmd, sess);
    add_upstream_native_libraries(cmd, sess);

    // # Telling the linker what we're doing

    if dylib {
        cmd.arg("/DLL");
    }

    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // addl_lib_search_paths
    if sess.opts.cg.rpath {
        let sysroot = sess.sysroot();
        let target_triple = &sess.opts.target_triple[];
        let get_install_prefix_lib_path = || {
            let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");
            let tlib = filesearch::relative_target_lib_path(sysroot, target_triple);
            let mut path = Path::new(install_prefix);
            path.push(&tlib);

            path
        };
        let rpath_config = RPathConfig {
            used_crates: sess.cstore.get_used_crates(cstore::RequireDynamic),
            out_filename: out_filename.clone(),
            has_rpath: sess.target.target.options.has_rpath,
            is_like_osx: sess.target.target.options.is_like_osx,
            get_install_prefix_lib_path: get_install_prefix_lib_path,
            realpath: ::util::fs::realpath
        };
        cmd.args(&rpath::get_rpath_flags(rpath_config)[]);
    }

    // Finally add all the linker arguments provided on the command line along
    // with any #[link_args] attributes found inside the crate
    let empty = Vec::new();
    cmd.args(&sess.opts.cg.link_args.as_ref().unwrap_or(&empty)[]);
    cmd.args(&used_link_args[..]);
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
fn add_local_native_libraries(cmd: &mut Command, sess: &Session) {
    sess.target_filesearch(PathKind::All).for_each_lib_search_path(|path, _k| {
        path.as_str().map(|s| cmd.arg(format!("/LIBPATH:{}", s)));
        FileDoesntMatch
    });

    let libs = sess.cstore.get_used_libraries();
    let libs = libs.borrow();

    let staticlibs = libs.iter().filter_map(|&(ref l, kind)| {
        if kind == cstore::NativeStatic {Some(l)} else {None}
    });
    let others = libs.iter().filter(|&&(_, kind)| {
        kind != cstore::NativeStatic
    });

    let search_path = super::link::archive_search_paths(sess);
    for l in staticlibs {
        let lib = archive::find_library(&l[..],
                                        &sess.target.target.options.staticlib_prefix,
                                        &sess.target.target.options.staticlib_suffix,
                                        &search_path[..],
                                        &sess.diagnostic().handler);
        let mut v = Vec::new();
        v.push_all(lib.as_vec());
        cmd.arg(&v[..]);
    }

    for &(ref l, kind) in others {
        match kind {
            cstore::NativeUnknown => {
                cmd.arg(format!("{}.lib", l));
            }
            cstore::NativeFramework => {}
            cstore::NativeStatic => unreachable!(),
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
fn add_upstream_rust_crates(cmd: &mut Command, sess: &Session,
                            dylib: bool, tmpdir: &Path,
                            trans: &CrateTranslation) {
    // All of the heavy lifting has previously been accomplished by the
    // dependency_format module of the compiler. This is just crawling the
    // output of that module, adding crates as necessary.
    //
    // Linking to a rlib involves just passing it to the linker (the linker
    // will slurp up the object files inside), and linking to a dynamic library
    // involves just passing the right -l flag.

    let data = if dylib {
        &trans.crate_formats[config::CrateTypeDylib]
    } else {
        &trans.crate_formats[config::CrateTypeExecutable]
    };

    // Invoke get_used_crates to ensure that we get a topological sorting of
    // crates.
    let deps = sess.cstore.get_used_crates(cstore::RequireDynamic);

    for &(cnum, _) in &deps {
        // We may not pass all crates through to the linker. Some crates may
        // appear statically in an existing dylib, meaning we'll pick up all the
        // symbols from the dylib.
        let kind = match data[cnum as uint - 1] {
            Some(t) => t,
            None => continue
        };
        let src = sess.cstore.get_used_crate_source(cnum).unwrap();
        match kind {
            cstore::RequireDynamic => {
                add_dynamic_crate(cmd, sess, src.dylib.unwrap().0)
            }
            cstore::RequireStatic => {
                add_static_crate(cmd, sess, tmpdir, src.rlib.unwrap().0)
            }
        }

    }

    // Converts a library file-stem into a cc -l argument
    fn unlib<'a>(config: &config::Config, stem: &'a [u8]) -> &'a [u8] {
        if stem.starts_with("lib".as_bytes()) && !config.target.options.is_like_windows {
            &stem[3..]
        } else {
            stem
        }
    }

    // Adds the static "rlib" versions of all crates to the command line.
    fn add_static_crate(cmd: &mut Command, sess: &Session, tmpdir: &Path,
                        cratepath: Path) {
        // When performing LTO on an executable output, all of the
        // bytecode from the upstream libraries has already been
        // included in our object file output. We need to modify all of
        // the upstream archives to remove their corresponding object
        // file to make sure we don't pull the same code in twice.
        //
        // We must continue to link to the upstream archives to be sure
        // to pull in native static dependencies. As the final caveat,
        // on Linux it is apparently illegal to link to a blank archive,
        // so if an archive no longer has any object files in it after
        // we remove `lib.o`, then don't link against it at all.
        //
        // If we're not doing LTO, then our job is simply to just link
        // against the archive.
        if sess.lto() {
            let name = cratepath.filename_str().unwrap();
            let name = &name[3..name.len() - 5]; // chop off lib/.rlib
            time(sess.time_passes(),
                 &format!("altering {}.rlib", name)[],
                 (), |()| {
                let dst = tmpdir.join(cratepath.filename().unwrap());
                match fs::copy(&cratepath, &dst) {
                    Ok(..) => {}
                    Err(e) => {
                        sess.err(&format!("failed to copy {} to {}: {}",
                                         cratepath.display(),
                                         dst.display(),
                                         e)[]);
                        sess.abort_if_errors();
                    }
                }
                // Fix up permissions of the copy, as fs::copy() preserves
                // permissions, but the original file may have been installed
                // by a package manager and may be read-only.
                match fs::chmod(&dst, old_io::USER_READ | old_io::USER_WRITE) {
                    Ok(..) => {}
                    Err(e) => {
                        sess.err(&format!("failed to chmod {} when preparing \
                                          for LTO: {}", dst.display(),
                                         e)[]);
                        sess.abort_if_errors();
                    }
                }
                let handler = &sess.diagnostic().handler;
                let config = ArchiveConfig {
                    handler: handler,
                    dst: dst.clone(),
                    lib_search_paths: super::link::archive_search_paths(sess),
                    slib_prefix: sess.target.target.options.staticlib_prefix.clone(),
                    slib_suffix: sess.target.target.options.staticlib_suffix.clone(),
                    maybe_ar_prog: sess.opts.cg.ar.clone()
                };
                let mut archive = Archive::open(config);
                archive.remove_file(&format!("{}.o", name)[]);
                let files = archive.files();
                if files.iter().any(|s| s[].ends_with(".o")) {
                    cmd.arg(dst);
                }
            });
        } else {
            cmd.arg(cratepath);
        }
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crate(cmd: &mut Command, sess: &Session, cratepath: Path) {
        // If we're performing LTO, then it should have been previously required
        // that all upstream rust dependencies were available in an rlib format.
        assert!(!sess.lto());

        cratepath.as_str().map(|s| {
            let libname = s.replace(".dll", ".lib");
            cmd.arg(&libname[]);
        });
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
fn add_upstream_native_libraries(cmd: &mut Command, sess: &Session) {
    // Be sure to use a topological sorting of crates because there may be
    // interdependencies between native libraries. When passing -nodefaultlibs,
    // for example, almost all native libraries depend on libc, so we have to
    // make sure that's all the way at the right (liblibc is near the base of
    // the dependency chain).
    //
    // This passes RequireStatic, but the actual requirement doesn't matter,
    // we're just getting an ordering of crate numbers, we're not worried about
    // the paths.
    let crates = sess.cstore.get_used_crates(cstore::RequireStatic);
    for (cnum, _) in crates {
        let libs = csearch::get_native_libraries(&sess.cstore, cnum);
        for &(kind, ref lib) in &libs {
            match kind {
                cstore::NativeUnknown => {
                    cmd.arg(format!("{}.lib", lib));
                }
                cstore::NativeFramework => {
                }
                cstore::NativeStatic => {
                    sess.bug("statics shouldn't be propagated");
                }
            }
        }
    }
}
