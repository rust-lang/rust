use std::env;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use tempfile::Builder as TempFileBuilder;

use rustc::session::config::{self, CrateType, DebugInfo, RUST_CGU_EXT};
use rustc::session::search_paths::PathKind;
use rustc::session::Session;
use rustc_codegen_utils::command::Command;
use rustc_codegen_utils::linker::*;
use rustc_fs_util::fix_windows_verbatim_for_gcc;
use rustc_target::spec::{LinkerFlavor, PanicStrategy, RelroLevel};

use crate::prelude::*;

use crate::link_copied::*;

pub(crate) fn link_rlib(sess: &Session, res: &CodegenResults, output_name: PathBuf) {
    let file = File::create(&output_name).unwrap();
    let mut builder = ar::Builder::new(file);

    // Add main object file
    let obj = res.artifact.emit().unwrap();
    builder
        .append(
            &ar::Header::new(b"data.o".to_vec(), obj.len() as u64),
            ::std::io::Cursor::new(obj),
        )
        .unwrap();

    // Non object files need to be added after object files, because ranlib will
    // try to read the native architecture from the first file, even if it isn't
    // an object file
    builder
        .append(
            &ar::Header::new(
                crate::metadata::METADATA_FILENAME.as_bytes().to_vec(),
                res.metadata.len() as u64,
            ),
            ::std::io::Cursor::new(res.metadata.clone()),
        )
        .unwrap();

    // Finalize archive
    std::mem::drop(builder);

    // Run ranlib to be able to link the archive
    let status = std::process::Command::new("ranlib")
        .arg(output_name)
        .status()
        .expect("Couldn't run ranlib");
    if !status.success() {
        sess.fatal(&format!("Ranlib exited with code {:?}", status.code()));
    }
}

pub(crate) fn link_bin(sess: &Session, codegen_results: &CodegenResults, out_filename: &Path) {
    let tmpdir = match TempFileBuilder::new().prefix("rustc").tempdir() {
        Ok(tmpdir) => tmpdir,
        Err(err) => sess.fatal(&format!("couldn't create a temp dir: {}", err)),
    };

    // TODO: link executable
    let obj = codegen_results.artifact.emit().unwrap();
    std::fs::write(tmpdir.path().join("out".to_string() + RUST_CGU_EXT), obj).unwrap();

    let (linker, flavor) = linker_and_flavor(sess);
    let (pname, mut cmd) = get_linker(sess, &linker, flavor);

    let root = sess.target_filesearch(PathKind::Native).get_lib_path();
    if let Some(args) = sess.target.target.options.pre_link_args.get(&flavor) {
        cmd.args(args);
    }
    if let Some(args) = sess.target.target.options.pre_link_args_crt.get(&flavor) {
        if sess.crt_static() {
            cmd.args(args);
        }
    }
    if let Some(ref args) = sess.opts.debugging_opts.pre_link_args {
        cmd.args(args);
    }
    cmd.args(&sess.opts.debugging_opts.pre_link_arg);

    for obj in &sess.target.target.options.pre_link_objects_exe {
        cmd.arg(root.join(obj));
    }

    if sess.crt_static() {
        for obj in &sess.target.target.options.pre_link_objects_exe_crt {
            cmd.arg(root.join(obj));
        }
    }

    if sess.target.target.options.is_like_emscripten {
        cmd.arg("-s");
        cmd.arg(if sess.panic_strategy() == PanicStrategy::Abort {
            "DISABLE_EXCEPTION_CATCHING=1"
        } else {
            "DISABLE_EXCEPTION_CATCHING=0"
        });
    }

    {
        let target_cpu = "x86_64-apple-darwin"; //::llvm_util::target_cpu(sess);
        let mut linker = codegen_results.linker_info.to_linker(cmd, &sess, flavor, target_cpu);
        link_args(&mut *linker, flavor, sess, CrateType::Executable, tmpdir.path(),
                  out_filename, codegen_results);
        cmd = linker.finalize();
    }
    if let Some(args) = sess.target.target.options.late_link_args.get(&flavor) {
        cmd.args(args);
    }
    for obj in &sess.target.target.options.post_link_objects {
        cmd.arg(root.join(obj));
    }
    if sess.crt_static() {
        for obj in &sess.target.target.options.post_link_objects_crt {
            cmd.arg(root.join(obj));
        }
    }
    if let Some(args) = sess.target.target.options.post_link_args.get(&flavor) {
        cmd.args(args);
    }
    for &(ref k, ref v) in &sess.target.target.options.link_env {
        cmd.env(k, v);
    }

    if sess.opts.debugging_opts.print_link_args {
        println!("{:?}", &cmd);
    }

    // May have not found libraries in the right formats.
    sess.abort_if_errors();

    // Invoke the system linker
    //
    // Note that there's a terribly awful hack that really shouldn't be present
    // in any compiler. Here an environment variable is supported to
    // automatically retry the linker invocation if the linker looks like it
    // segfaulted.
    //
    // Gee that seems odd, normally segfaults are things we want to know about!
    // Unfortunately though in rust-lang/rust#38878 we're experiencing the
    // linker segfaulting on Travis quite a bit which is causing quite a bit of
    // pain to land PRs when they spuriously fail due to a segfault.
    //
    // The issue #38878 has some more debugging information on it as well, but
    // this unfortunately looks like it's just a race condition in macOS's linker
    // with some thread pool working in the background. It seems that no one
    // currently knows a fix for this so in the meantime we're left with this...
    let retry_on_segfault = env::var("RUSTC_RETRY_LINKER_ON_SEGFAULT").is_ok();
    let mut prog;
    let mut i = 0;
    loop {
        i += 1;
        prog = exec_linker(sess, &mut cmd, out_filename, tmpdir.path());
        let output = match prog {
            Ok(ref output) => output,
            Err(_) => break,
        };
        if output.status.success() {
            break
        }
        let mut out = output.stderr.clone();
        out.extend(&output.stdout);
        let out = String::from_utf8_lossy(&out);

        // Check to see if the link failed with "unrecognized command line option:
        // '-no-pie'" for gcc or "unknown argument: '-no-pie'" for clang. If so,
        // reperform the link step without the -no-pie option. This is safe because
        // if the linker doesn't support -no-pie then it should not default to
        // linking executables as pie. Different versions of gcc seem to use
        // different quotes in the error message so don't check for them.
        if sess.target.target.options.linker_is_gnu &&
           flavor != LinkerFlavor::Ld &&
           (out.contains("unrecognized command line option") ||
            out.contains("unknown argument")) &&
           out.contains("-no-pie") &&
           cmd.get_args().iter().any(|e| e.to_string_lossy() == "-no-pie") {
            for arg in cmd.take_args() {
                if arg.to_string_lossy() != "-no-pie" {
                    cmd.arg(arg);
                }
            }
            continue;
        }
        if !retry_on_segfault || i > 3 {
            break
        }
        let msg_segv = "clang: error: unable to execute command: Segmentation fault: 11";
        let msg_bus  = "clang: error: unable to execute command: Bus error: 10";
        if !(out.contains(msg_segv) || out.contains(msg_bus)) {
            break
        }
    }

    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                sess.struct_err(&format!("linking with `{}` failed: {}",
                                         pname.display(),
                                         prog.status))
                    .note(&format!("{:?}", &cmd))
                    .note(&String::from_utf8_lossy(&output))
                    .emit();
                sess.abort_if_errors();
            }
        },
        Err(e) => {
            let linker_not_found = e.kind() == io::ErrorKind::NotFound;

            let mut linker_error = {
                if linker_not_found {
                    sess.struct_err(&format!("linker `{}` not found", pname.display()))
                } else {
                    sess.struct_err(&format!("could not exec the linker `{}`", pname.display()))
                }
            };

            linker_error.note(&e.to_string());

            if !linker_not_found {
                linker_error.note(&format!("{:?}", &cmd));
            }

            linker_error.emit();

            if sess.target.target.options.is_like_msvc && linker_not_found {
                sess.note_without_error("the msvc targets depend on the msvc linker \
                    but `link.exe` was not found");
                sess.note_without_error("please ensure that VS 2013, VS 2015 or VS 2017 \
                    was installed with the Visual C++ option");
            }
            sess.abort_if_errors();
        }
    }


    // On macOS, debuggers need this utility to get run to do some munging of
    // the symbols. Note, though, that if the object files are being preserved
    // for their debug information there's no need for us to run dsymutil.
    if sess.target.target.options.is_like_osx &&
        sess.opts.debuginfo != DebugInfo::None
    {
        match Command::new("dsymutil").arg(out_filename).output() {
            Ok(..) => {}
            Err(e) => sess.fatal(&format!("failed to run dsymutil: {}", e)),
        }
    }
}

/*
res.artifact
    .declare_with(
        &metadata_name,
        faerie::artifact::Decl::Data {
            global: true,
            writable: false,
        },
        res.metadata.clone(),
    )
    .unwrap();
*/


fn link_args(cmd: &mut dyn Linker,
             flavor: LinkerFlavor,
             sess: &Session,
             crate_type: config::CrateType,
             tmpdir: &Path,
             out_filename: &Path,
             codegen_results: &CodegenResults) {

    // Linker plugins should be specified early in the list of arguments
    cmd.cross_lang_lto();

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = sess.target_filesearch(PathKind::All).get_lib_path();

    // target descriptor
    let t = &sess.target.target;

    cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));
    for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
        cmd.add_object(obj);
    }
    cmd.output_filename(out_filename);

    // If we're building a dynamic library then some platforms need to make sure
    // that all symbols are exported correctly from the dynamic library.
    if crate_type != config::CrateType::Executable ||
       sess.target.target.options.is_like_emscripten {
        cmd.export_symbols(tmpdir, crate_type);
    }

    let obj = codegen_results.allocator_module
        .as_ref()
        .and_then(|m| m.object.as_ref());
    if let Some(obj) = obj {
        cmd.add_object(obj);
    }

    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    if !sess.opts.cg.link_dead_code {
        let keep_metadata = crate_type == config::CrateType::Dylib;
        cmd.gc_sections(keep_metadata);
    }

    let used_link_args = &codegen_results.crate_info.link_args;

    if crate_type == config::CrateType::Executable {
        let mut position_independent_executable = false;

        if t.options.position_independent_executables {
            let empty_vec = Vec::new();
            let args = sess.opts.cg.link_args.as_ref().unwrap_or(&empty_vec);
            let more_args = &sess.opts.cg.link_arg;
            let mut args = args.iter().chain(more_args.iter()).chain(used_link_args.iter());

            if !sess.crt_static() && !args.any(|x| *x == "-static") {
                position_independent_executable = true;
            }
        }

        if position_independent_executable {
            cmd.position_independent_executable();
        } else {
            // recent versions of gcc can be configured to generate position
            // independent executables by default. We have to pass -no-pie to
            // explicitly turn that off. Not applicable to ld.
            if sess.target.target.options.linker_is_gnu
                && flavor != LinkerFlavor::Ld {
                cmd.no_position_independent_executable();
            }
        }
    }

    let relro_level = match sess.opts.debugging_opts.relro_level {
        Some(level) => level,
        None => t.options.relro_level,
    };
    match relro_level {
        RelroLevel::Full => {
            cmd.full_relro();
        },
        RelroLevel::Partial => {
            cmd.partial_relro();
        },
        RelroLevel::Off => {
            cmd.no_relro();
        },
        RelroLevel::None => {
        },
    }

    // Pass optimization flags down to the linker.
    cmd.optimize();

    // Pass debuginfo flags down to the linker.
    cmd.debuginfo();

    // We want to, by default, prevent the compiler from accidentally leaking in
    // any system libraries, so we may explicitly ask linkers to not link to any
    // libraries by default. Note that this does not happen for windows because
    // windows pulls in some large number of libraries and I couldn't quite
    // figure out which subset we wanted.
    //
    // This is all naturally configurable via the standard methods as well.
    if !sess.opts.cg.default_linker_libraries.unwrap_or(false) &&
        t.options.no_default_libraries
    {
        cmd.no_default_libraries();
    }

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
    //  2. Local native libraries
    //  3. Upstream rust libraries
    //  4. Upstream native libraries
    //
    // The rationale behind this ordering is that those items lower down in the
    // list can't depend on items higher up in the list. For example nothing can
    // depend on what we just generated (e.g. that'd be a circular dependency).
    // Upstream rust libraries are not allowed to depend on our local native
    // libraries as that would violate the structure of the DAG, in that
    // scenario they are required to link to them as well in a shared fashion.
    //
    // Note that upstream rust libraries may contain native dependencies as
    // well, but they also can't depend on what we just started to add to the
    // link line. And finally upstream native libraries can't depend on anything
    // in this DAG so far because they're only dylibs and dylibs can only depend
    // on other dylibs (e.g. other native deps).
    add_local_native_libraries(cmd, sess, codegen_results);
    add_upstream_rust_crates(cmd, sess, codegen_results, crate_type, tmpdir);
    add_upstream_native_libraries(cmd, sess, codegen_results, crate_type);

    // Tell the linker what we're doing.
    if crate_type != config::CrateType::Executable {
        cmd.build_dylib(out_filename);
    }
    if crate_type == config::CrateType::Executable && sess.crt_static() {
        cmd.build_static_executable();
    }

    if sess.opts.debugging_opts.pgo_gen.is_some() {
        cmd.pgo_gen();
    }

    // Finally add all the linker arguments provided on the command line along
    // with any #[link_args] attributes found inside the crate
    if let Some(ref args) = sess.opts.cg.link_args {
        cmd.args(args);
    }
    cmd.args(&sess.opts.cg.link_arg);
    cmd.args(&used_link_args);
}
