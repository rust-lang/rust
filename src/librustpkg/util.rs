// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc;
use std::os;
use std::io;
use std::io::fs;
use extra::workcache;
use rustc::driver::{driver, session};
use extra::getopts::groups::getopts;
use syntax::ast_util::*;
use syntax::codemap::{dummy_sp, Spanned};
use syntax::ext::base::ExtCtxt;
use syntax::{ast, attr, codemap, diagnostic, fold, visit};
use syntax::attr::AttrMetaMethods;
use syntax::fold::ast_fold;
use syntax::visit::Visitor;
use rustc::back::link::output_type_exe;
use rustc::back::link;
use rustc::driver::session::{lib_crate, bin_crate};
use context::{in_target, StopBefore, Link, Assemble, BuildContext};
use package_id::PkgId;
use package_source::PkgSrc;
use workspace::pkg_parent_workspaces;
use path_util::{system_library, target_build_dir};
use path_util::{default_workspace, built_library_in_workspace};
pub use target::{OutputType, Main, Lib, Bench, Test, JustOne, lib_name_of, lib_crate_filename};
pub use target::{Target, Build, Install};
use extra::treemap::TreeMap;
pub use target::{lib_name_of, lib_crate_filename, WhatToBuild, MaybeCustom, Inferred};
use workcache_support::{digest_file_with_date, digest_only_date};
use messages::error;

// It would be nice to have the list of commands in just one place -- for example,
// you could update the match in rustpkg.rc but forget to update this list. I think
// that should be fixed.
static COMMANDS: &'static [&'static str] =
    &["build", "clean", "do", "info", "init", "install", "list", "prefer", "test", "uninstall",
      "unprefer"];


pub type ExitCode = int; // For now

pub struct Pkg {
    id: PkgId,
    bins: ~[~str],
    libs: ~[~str],
}

impl ToStr for Pkg {
    fn to_str(&self) -> ~str {
        self.id.to_str()
    }
}

pub fn is_cmd(cmd: &str) -> bool {
    COMMANDS.iter().any(|&c| c == cmd)
}

struct ListenerFn {
    cmds: ~[~str],
    span: codemap::Span,
    path: ~[ast::Ident]
}

struct ReadyCtx {
    sess: session::Session,
    ext_cx: @ExtCtxt,
    path: ~[ast::Ident],
    fns: ~[ListenerFn]
}

fn fold_mod(_ctx: @mut ReadyCtx, m: &ast::_mod, fold: &CrateSetup)
            -> ast::_mod {
    fn strip_main(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: do item.attrs.iter().filter_map |attr| {
                if "main" != attr.name() {
                    Some(*attr)
                } else {
                    None
                }
            }.collect(),
            .. (*item).clone()
        }
    }

    fold::noop_fold_mod(&ast::_mod {
        items: do m.items.map |item| {
            strip_main(*item)
        },
        .. (*m).clone()
    }, fold)
}

fn fold_item(ctx: @mut ReadyCtx, item: @ast::item, fold: &CrateSetup)
             -> Option<@ast::item> {
    ctx.path.push(item.ident);

    let mut cmds = ~[];
    let mut had_pkg_do = false;

    for attr in item.attrs.iter() {
        if "pkg_do" == attr.name() {
            had_pkg_do = true;
            match attr.node.value.node {
                ast::MetaList(_, ref mis) => {
                    for mi in mis.iter() {
                        match mi.node {
                            ast::MetaWord(cmd) => cmds.push(cmd.to_owned()),
                            _ => {}
                        };
                    }
                }
                _ => cmds.push(~"build")
            }
        }
    }

    if had_pkg_do {
        ctx.fns.push(ListenerFn {
            cmds: cmds,
            span: item.span,
            path: /*bad*/ctx.path.clone()
        });
    }

    let res = fold::noop_fold_item(item, fold);

    ctx.path.pop();

    res
}

struct CrateSetup {
    ctx: @mut ReadyCtx,
}

impl fold::ast_fold for CrateSetup {
    fn fold_item(&self, item: @ast::item) -> Option<@ast::item> {
        fold_item(self.ctx, item, self)
    }
    fn fold_mod(&self, module: &ast::_mod) -> ast::_mod {
        fold_mod(self.ctx, module, self)
    }
}

/// Generate/filter main function, add the list of commands, etc.
pub fn ready_crate(sess: session::Session,
                   crate: ast::Crate) -> ast::Crate {
    let ctx = @mut ReadyCtx {
        sess: sess,
        ext_cx: ExtCtxt::new(sess.parse_sess, sess.opts.cfg.clone()),
        path: ~[],
        fns: ~[]
    };
    let fold = CrateSetup {
        ctx: ctx,
    };
    fold.fold_crate(crate)
}

pub fn compile_input(context: &BuildContext,
                     exec: &mut workcache::Exec,
                     pkg_id: &PkgId,
                     in_file: &Path,
                     workspace: &Path,
                     deps: &mut DepMap,
                     flags: &[~str],
                     cfgs: &[~str],
                     opt: session::OptLevel,
                     what: OutputType) -> Option<Path> {
    assert!(in_file.components().nth(1).is_some());
    let input = driver::file_input(in_file.clone());
    debug!("compile_input: {} / {:?}", in_file.display(), what);
    // tjc: by default, use the package ID name as the link name
    // not sure if we should support anything else

    let mut out_dir = target_build_dir(workspace);
    out_dir.push(&pkg_id.path);
    // Make the output directory if it doesn't exist already
    fs::mkdir_recursive(&out_dir, io::UserRWX);

    let binary = os::args()[0].to_managed();

    debug!("flags: {}", flags.connect(" "));
    debug!("cfgs: {}", cfgs.connect(" "));
    let csysroot = context.sysroot();
    debug!("compile_input's sysroot = {}", csysroot.display());

    let crate_type = match what {
        Lib => lib_crate,
        Test | Bench | Main => bin_crate
    };
    let matches = getopts(debug_flags()
                          + match what {
                              Lib => ~[~"--lib"],
                              // --test compiles both #[test] and #[bench] fns
                              Test | Bench => ~[~"--test"],
                              Main => ~[]
                          }
                          + flags
                          + context.flag_strs()
                          + cfgs.flat_map(|c| { ~[~"--cfg", (*c).clone()] }),
                          driver::optgroups()).unwrap();
    debug!("rustc flags: {:?}", matches);

    // Hack so that rustpkg can run either out of a rustc target dir,
    // or the host dir
    let sysroot_to_use = @if !in_target(&context.sysroot()) {
        context.sysroot()
    }
    else {
        let mut p = context.sysroot().clone();
        p.pop();
        p.pop();
        p.pop();
        p
    };
    let csysroot = context.sysroot();
    debug!("compile_input's sysroot = {}", csysroot.display());
    debug!("sysroot_to_use = {}", sysroot_to_use.display());

    let output_type = match context.compile_upto() {
        Assemble => link::output_type_assembly,
        Link     => link::output_type_object,
        Pretty | Trans | Analysis => link::output_type_none,
        LLVMAssemble => link::output_type_llvm_assembly,
        LLVMCompileBitcode => link::output_type_bitcode,
        Nothing => link::output_type_exe
    };

    debug!("Output type = {:?}", output_type);

    let options = @session::options {
        crate_type: crate_type,
        optimize: opt,
        test: what == Test || what == Bench,
        maybe_sysroot: Some(sysroot_to_use),
        addl_lib_search_paths: @mut context.additional_library_paths(),
        output_type: output_type,
        .. (*driver::build_session_options(binary,
                                           &matches,
                                           @diagnostic::DefaultEmitter as
                                            @diagnostic::Emitter)).clone()
    };

    debug!("Created options...");

    let addl_lib_search_paths = @mut options.addl_lib_search_paths;
    // Make sure all the library directories actually exist, since the linker will complain
    // otherwise
    for p in addl_lib_search_paths.iter() {
        if p.exists() {
            assert!(p.is_dir())
        }
        else {
            fs::mkdir_recursive(p, io::UserRWX);
        }
    }

    debug!("About to build session...");

    let sess = driver::build_session(options,
                                     @diagnostic::DefaultEmitter as
                                        @diagnostic::Emitter);

    debug!("About to build config...");

    // Infer dependencies that rustpkg needs to build, by scanning for
    // `extern mod` directives.
    let cfg = driver::build_configuration(sess);
    let mut crate = driver::phase_1_parse_input(sess, cfg.clone(), &input);
    crate = driver::phase_2_configure_and_expand(sess, cfg.clone(), crate);

    debug!("About to call find_and_install_dependencies...");

    find_and_install_dependencies(context, pkg_id, in_file, sess, exec, &crate, deps,
                                  |p| {
                                      debug!("a dependency: {}", p.display());
                                      // Pass the directory containing a dependency
                                      // as an additional lib search path
                                      addl_lib_search_paths.insert(p);
                                  });

    // Inject the link attributes so we get the right package name and version
    if attr::find_linkage_metas(crate.attrs).is_empty() {
        let name_to_use = match what {
            Test  => format!("{}test", pkg_id.short_name).to_managed(),
            Bench => format!("{}bench", pkg_id.short_name).to_managed(),
            _     => pkg_id.short_name.to_managed()
        };
        debug!("Injecting link name: {}", name_to_use);
        // FIXME (#9639): This needs to handle non-utf8 paths
        let link_options =
            ~[attr::mk_name_value_item_str(@"name", name_to_use),
              attr::mk_name_value_item_str(@"vers", pkg_id.version.to_str().to_managed())] +
            ~[attr::mk_name_value_item_str(@"package_id",
                                           pkg_id.path.as_str().unwrap().to_managed())];

        debug!("link options: {:?}", link_options);
        crate.attrs = ~[attr::mk_attr(attr::mk_list_item(@"link", link_options))];
    }

    debug!("calling compile_crate_from_input, workspace = {},
           building_library = {:?}", out_dir.display(), sess.building_library);
    let result = compile_crate_from_input(in_file,
                                          exec,
                                          context.compile_upto(),
                                          &out_dir,
                                          sess,
                                          crate);
    // Discover the output
    let discovered_output = if what == Lib  {
        built_library_in_workspace(pkg_id, workspace) // Huh???
    }
    else {
        result
    };
    for p in discovered_output.iter() {
        debug!("About to discover output {}", p.display());
        if p.exists() {
            debug!("4. discovering output {}", p.display());
            // FIXME (#9639): This needs to handle non-utf8 paths
            exec.discover_output("binary", p.as_str().unwrap(), digest_only_date(p));
        }
        // Nothing to do if it doesn't exist -- that could happen if we had the
        // -S or -emit-llvm flags, etc.
    }
    discovered_output
}

// Should use workcache to avoid recompiling when not necessary
// Should also rename this to something better
// If crate_opt is present, then finish compilation. If it's None, then
// call compile_upto and return the crate
// also, too many arguments
// Returns list of discovered dependencies
pub fn compile_crate_from_input(input: &Path,
                                exec: &mut workcache::Exec,
                                stop_before: StopBefore,
 // should be of the form <workspace>/build/<pkg id's path>
                                out_dir: &Path,
                                sess: session::Session,
// Returns None if one of the flags that suppresses compilation output was
// given
                                crate: ast::Crate) -> Option<Path> {
    debug!("Calling build_output_filenames with {}, building library? {:?}",
           out_dir.display(), sess.building_library);

    // bad copy
    debug!("out_dir = {}", out_dir.display());
    let outputs = driver::build_output_filenames(&driver::file_input(input.clone()),
                                                 &Some(out_dir.clone()), &None,
                                                 crate.attrs, sess);

    debug!("Outputs are out_filename: {} and obj_filename: {} and output type = {:?}",
           outputs.out_filename.display(),
           outputs.obj_filename.display(),
           sess.opts.output_type);
    debug!("additional libraries:");
    for lib in sess.opts.addl_lib_search_paths.iter() {
        debug!("an additional library: {}", lib.display());
    }
    let analysis = driver::phase_3_run_analysis_passes(sess, &crate);
    if driver::stop_after_phase_3(sess) { return None; }
    let translation = driver::phase_4_translate_to_llvm(sess, crate,
                                                        &analysis,
                                                        outputs);
    driver::phase_5_run_llvm_passes(sess, &translation, outputs);
    // The second check shouldn't be necessary, but rustc seems to ignore
    // -c
    if driver::stop_after_phase_5(sess)
        || stop_before == Link || stop_before == Assemble { return Some(outputs.out_filename); }
    driver::phase_6_link_output(sess, &translation, outputs);

    // Register dependency on the source file
    // FIXME (#9639): This needs to handle non-utf8 paths
    exec.discover_input("file", input.as_str().unwrap(), digest_file_with_date(input));

    debug!("Built {}, date = {:?}", outputs.out_filename.display(),
           datestamp(&outputs.out_filename));
    Some(outputs.out_filename)
}

#[cfg(windows)]
pub fn exe_suffix() -> ~str { ~".exe" }

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn exe_suffix() -> ~str { ~"" }

// Called by build_crates
pub fn compile_crate(ctxt: &BuildContext,
                     exec: &mut workcache::Exec,
                     pkg_id: &PkgId,
                     crate: &Path,
                     workspace: &Path,
                     deps: &mut DepMap,
                     flags: &[~str],
                     cfgs: &[~str],
                     opt: session::OptLevel,
                     what: OutputType) -> Option<Path> {
    debug!("compile_crate: crate={}, workspace={}", crate.display(), workspace.display());
    debug!("compile_crate: short_name = {}, flags =...", pkg_id.to_str());
    for fl in flags.iter() {
        debug!("+++ {}", *fl);
    }
    compile_input(ctxt, exec, pkg_id, crate, workspace, deps, flags, cfgs, opt, what)
}

struct ViewItemVisitor<'self> {
    context: &'self BuildContext,
    parent: &'self PkgId,
    parent_crate: &'self Path,
    sess: session::Session,
    exec: &'self mut workcache::Exec,
    c: &'self ast::Crate,
    save: &'self fn(Path),
    deps: &'self mut DepMap
}

impl<'self> Visitor<()> for ViewItemVisitor<'self> {
    fn visit_view_item(&mut self, vi: &ast::view_item, env: ()) {
        use conditions::nonexistent_package::cond;

        match vi.node {
            // ignore metadata, I guess
            ast::view_item_extern_mod(lib_ident, path_opt, _, _) => {
                let lib_name = match path_opt {
                    Some((p, _)) => p,
                    None => self.sess.str_of(lib_ident)
                };
                debug!("Finding and installing... {}", lib_name);
                // Check standard Rust library path first
                let whatever = system_library(&self.context.sysroot(), lib_name);
                debug!("system library returned {:?}", whatever);
                match whatever {
                    Some(ref installed_path) => {
                        debug!("It exists: {}", installed_path.display());
                        // Say that [path for c] has a discovered dependency on
                        // installed_path
                        // For binary files, we only hash the datestamp, not the contents.
                        // I'm not sure what the right thing is.
                        // Now we know that this crate has a discovered dependency on
                        // installed_path
                        // FIXME (#9639): This needs to handle non-utf8 paths
                        add_dep(self.deps, self.parent_crate.as_str().unwrap().to_owned(),
                                (~"binary", installed_path.as_str().unwrap().to_owned()));
                        self.exec.discover_input("binary",
                                                 installed_path.as_str().unwrap(),
                                                 digest_only_date(installed_path));
                    }
                    None => {
                        // FIXME #8711: need to parse version out of path_opt
                        debug!("Trying to install library {}, rebuilding it",
                               lib_name.to_str());
                        // Try to install it
                        let pkg_id = PkgId::new(lib_name);
                        // Find all the workspaces in the RUST_PATH that contain this package.
                        let workspaces = pkg_parent_workspaces(&self.context.context,
                                                               &pkg_id);
                        // Three cases:
                        // (a) `workspaces` is empty. That means there's no local source
                        // for this package. In that case, we pass the default workspace
                        // into `PkgSrc::new`, so that if it exists as a remote repository,
                        // its sources will be fetched into it. We also put the output in the
                        // same workspace.
                        // (b) We're using the Rust path hack. In that case, the output goes
                        // in the destination workspace.
                        // (c) `workspaces` is non-empty -- we found a local source for this
                        // package and will build in that workspace.
                        let (source_workspace, dest_workspace) = if workspaces.is_empty() {
                            (default_workspace(), default_workspace())
                        } else {
                            if self.context.context.use_rust_path_hack {
                                (workspaces[0], default_workspace())
                            } else {
                                 (workspaces[0].clone(), workspaces[0])
                            }
                        };
                        // In this case, the source and destination workspaces are the same:
                        // Either it's a remote package, so the local sources don't exist
                        // and the `PkgSrc` constructor will detect that;
                        // or else it's already in a workspace and we'll build into that
                        // workspace
                        let pkg_src = do cond.trap(|_| {
                                 // Nonexistent package? Then print a better error
                                 error(format!("Package {} depends on {}, but I don't know \
                                               how to find it",
                                               self.parent.path.display(),
                                               pkg_id.path.display()));
                                 fail!()
                        }).inside {
                            PkgSrc::new(source_workspace.clone(),
                                        dest_workspace.clone(),
                                        // Use the rust_path_hack to search for dependencies iff
                                        // we were already using it
                                        self.context.context.use_rust_path_hack,
                                        pkg_id.clone())
                        };
                        let (outputs_disc, inputs_disc) =
                            self.context.install(
                                pkg_src,
                                &WhatToBuild::new(Inferred,
                                                  JustOne(Path::new(lib_crate_filename))));
                        debug!("Installed {}, returned {:?} dependencies and \
                               {:?} transitive dependencies",
                               lib_name, outputs_disc.len(), inputs_disc.len());
                        debug!("discovered outputs = {:?} discovered_inputs = {:?}",
                               outputs_disc, inputs_disc);
                        // It must have installed *something*...
                        assert!(!outputs_disc.is_empty());
                        for dep in outputs_disc.iter() {
                            debug!("Discovering a binary input: {}", dep.display());
                            // FIXME (#9639): This needs to handle non-utf8 paths
                            self.exec.discover_input("binary",
                                                     dep.as_str().unwrap(),
                                                     digest_only_date(dep));
                            add_dep(self.deps,
                                    self.parent_crate.as_str().unwrap().to_owned(),
                                    (~"binary", dep.as_str().unwrap().to_owned()));

                            // Also, add an additional search path
                            let dep_dir = dep.dir_path();
                            debug!("Installed {} into {}", dep.display(), dep_dir.display());
                            (self.save)(dep_dir);
                        }
                        debug!("Installed {}, returned {} dependencies and \
                                {} transitive dependencies",
                                lib_name, outputs_disc.len(), inputs_disc.len());
                        // It must have installed *something*...
                        assert!(!outputs_disc.is_empty());
                        let mut target_workspace = outputs_disc[0].clone();
                        target_workspace.pop();
                        for &(ref what, ref dep) in inputs_disc.iter() {
                            if *what == ~"file" {
                                add_dep(self.deps,
                                        self.parent_crate.as_str().unwrap().to_owned(),
                                        (~"file", dep.clone()));
                                self.exec.discover_input(*what,
                                                         *dep,
                                                         digest_file_with_date(
                                                             &Path::new(dep.as_slice())));
                            } else if *what == ~"binary" {
                                add_dep(self.deps,
                                        self.parent_crate.as_str().unwrap().to_owned(),
                                        (~"binary", dep.clone()));
                                self.exec.discover_input(*what,
                                                         *dep,
                                                         digest_only_date(
                                                             &Path::new(dep.as_slice())));
                            } else {
                                fail!("Bad kind: {}", *what);
                            }
                            // Also, add an additional search path
                            debug!("Installed {} into {}",
                                    lib_name, target_workspace.as_str().unwrap().to_owned());
                            (self.save)(target_workspace.clone());
                        }
                    }
                }
            }
            // Ignore `use`s
            _ => ()
        }
        visit::walk_view_item(self, vi, env)
    }
}

/// Collect all `extern mod` directives in `c`, then
/// try to install their targets, failing if any target
/// can't be found.
pub fn find_and_install_dependencies(context: &BuildContext,
                                     parent: &PkgId,
                                     parent_crate: &Path,
                                     sess: session::Session,
                                     exec: &mut workcache::Exec,
                                     c: &ast::Crate,
                                     deps: &mut DepMap,
                                     save: &fn(Path)) {
    debug!("In find_and_install_dependencies...");
    let mut visitor = ViewItemVisitor {
        context: context,
        parent: parent,
        parent_crate: parent_crate,
        sess: sess,
        exec: exec,
        c: c,
        save: save,
        deps: deps
    };
    visit::walk_crate(&mut visitor, c, ())
}

pub fn mk_string_lit(s: @str) -> ast::lit {
    Spanned {
        node: ast::lit_str(s, ast::CookedStr),
        span: dummy_sp()
    }
}

#[cfg(test)]
mod test {
    use super::is_cmd;

    #[test]
    fn test_is_cmd() {
        assert!(is_cmd("build"));
        assert!(is_cmd("clean"));
        assert!(is_cmd("do"));
        assert!(is_cmd("info"));
        assert!(is_cmd("install"));
        assert!(is_cmd("prefer"));
        assert!(is_cmd("test"));
        assert!(is_cmd("uninstall"));
        assert!(is_cmd("unprefer"));
    }

}

pub fn option_to_vec<T>(x: Option<T>) -> ~[T] {
    match x {
       Some(y) => ~[y],
       None    => ~[]
    }
}

// tjc: cheesy
fn debug_flags() -> ~[~str] { ~[] }
// static DEBUG_FLAGS: ~[~str] = ~[~"-Z", ~"time-passes"];


/// Returns the last-modified date as an Option
pub fn datestamp(p: &Path) -> Option<libc::time_t> {
    debug!("Scrutinizing datestamp for {} - does it exist? {:?}", p.display(),
           p.exists());
    match io::result(|| p.stat()) {
        Ok(s) => {
            let out = s.modified;
            debug!("Date = {:?}", out);
            Some(out as libc::time_t)
        }
        Err(*) => None,
    }
}

pub type DepMap = TreeMap<~str, ~[(~str, ~str)]>;

/// Records a dependency from `parent` to the kind and value described by `info`,
/// in `deps`
fn add_dep(deps: &mut DepMap, parent: ~str, info: (~str, ~str)) {
    let mut done = false;
    let info_clone = info.clone();
    match deps.find_mut(&parent) {
        None => { }
        Some(v) => { done = true; (*v).push(info) }
    };
    if !done {
        deps.insert(parent, ~[info_clone]);
    }
}
