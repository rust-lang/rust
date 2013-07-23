// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{os, result};
use rustc::driver::{driver, session};
use rustc::metadata::filesearch;
use extra::getopts::groups::getopts;
use syntax::ast_util::*;
use syntax::codemap::{dummy_sp, spanned};
use syntax::ext::base::ExtCtxt;
use syntax::{ast, attr, codemap, diagnostic, fold};
use syntax::attr::AttrMetaMethods;
use rustc::back::link::output_type_exe;
use rustc::driver::driver::compile_upto;
use rustc::driver::session::{lib_crate, bin_crate};
use context::Ctx;
use package_id::PkgId;
use search::find_library_in_search_path;
use path_util::target_library_in_workspace;
pub use target::{OutputType, Main, Lib, Bench, Test};

// It would be nice to have the list of commands in just one place -- for example,
// you could update the match in rustpkg.rc but forget to update this list. I think
// that should be fixed.
static COMMANDS: &'static [&'static str] =
    &["build", "clean", "do", "info", "install", "list", "prefer", "test", "uninstall",
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

pub fn root() -> Path {
    match filesearch::get_rustpkg_root() {
        result::Ok(path) => path,
        result::Err(err) => fail!(err)
    }
}

pub fn is_cmd(cmd: &str) -> bool {
    COMMANDS.iter().any(|&c| c == cmd)
}

struct ListenerFn {
    cmds: ~[~str],
    span: codemap::span,
    path: ~[ast::ident]
}

struct ReadyCtx {
    sess: session::Session,
    crate: @ast::Crate,
    ext_cx: @ExtCtxt,
    path: ~[ast::ident],
    fns: ~[ListenerFn]
}

fn fold_mod(_ctx: @mut ReadyCtx,
            m: &ast::_mod,
            fold: @fold::ast_fold) -> ast::_mod {
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

fn fold_item(ctx: @mut ReadyCtx,
             item: @ast::item,
             fold: @fold::ast_fold) -> Option<@ast::item> {
    ctx.path.push(item.ident);

    let mut cmds = ~[];
    let mut had_pkg_do = false;

    for item.attrs.iter().advance |attr| {
        if "pkg_do" == attr.name() {
            had_pkg_do = true;
            match attr.node.value.node {
                ast::MetaList(_, ref mis) => {
                    for mis.iter().advance |mi| {
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

/// Generate/filter main function, add the list of commands, etc.
pub fn ready_crate(sess: session::Session,
                   crate: @ast::Crate) -> @ast::Crate {
    let ctx = @mut ReadyCtx {
        sess: sess,
        crate: crate,
        ext_cx: ExtCtxt::new(sess.parse_sess, sess.opts.cfg.clone()),
        path: ~[],
        fns: ~[]
    };
    let precursor = @fold::AstFoldFns {
        // fold_crate: fold::wrap(|a, b| fold_crate(ctx, a, b)),
        fold_item: |a, b| fold_item(ctx, a, b),
        fold_mod: |a, b| fold_mod(ctx, a, b),
        .. *fold::default_ast_fold()
    };

    let fold = fold::make_fold(precursor);

    @fold.fold_crate(crate)
}

// FIXME (#4432): Use workcache to only compile when needed
pub fn compile_input(ctxt: &Ctx,
                     pkg_id: &PkgId,
                     in_file: &Path,
                     out_dir: &Path,
                     flags: &[~str],
                     cfgs: &[~str],
                     opt: bool,
                     what: OutputType) -> bool {

    let workspace = out_dir.pop().pop();

    assert!(in_file.components.len() > 1);
    let input = driver::file_input((*in_file).clone());
    debug!("compile_input: %s / %?", in_file.to_str(), what);
    // tjc: by default, use the package ID name as the link name
    // not sure if we should support anything else

    let binary = os::args()[0].to_managed();

    debug!("flags: %s", flags.connect(" "));
    debug!("cfgs: %s", cfgs.connect(" "));
    debug!("compile_input's sysroot = %?", ctxt.sysroot_opt);

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
                          + cfgs.flat_map(|c| { ~[~"--cfg", (*c).clone()] }),
                          driver::optgroups()).get();
    let options = @session::options {
        crate_type: crate_type,
        optimize: if opt { session::Aggressive } else { session::No },
        test: what == Test || what == Bench,
        maybe_sysroot: ctxt.sysroot_opt,
        addl_lib_search_paths: @mut (~[(*out_dir).clone()]),
        // output_type should be conditional
        output_type: output_type_exe, // Use this to get a library? That's weird
        .. (*driver::build_session_options(binary, &matches, diagnostic::emit)).clone()
    };

    let addl_lib_search_paths = @mut options.addl_lib_search_paths;
    // Make sure all the library directories actually exist, since the linker will complain
    // otherwise
    for addl_lib_search_paths.iter().advance |p| {
        assert!(os::path_is_dir(p));
    }

    let sess = driver::build_session(options, diagnostic::emit);

    // Infer dependencies that rustpkg needs to build, by scanning for
    // `extern mod` directives.
    let cfg = driver::build_configuration(sess, binary, &input);
    let (crate_opt, _) = driver::compile_upto(sess, cfg.clone(), &input, driver::cu_expand, None);

    let mut crate = match crate_opt {
        Some(c) => c,
        None => fail!("compile_input expected...")
    };

    // Not really right. Should search other workspaces too, and the installed
    // database (which doesn't exist yet)
    find_and_install_dependencies(ctxt, sess, &workspace, crate,
                                  |p| {
                                      debug!("a dependency: %s", p.to_str());
                                      // Pass the directory containing a dependency
                                      // as an additional lib search path
                                      addl_lib_search_paths.push(p);
                                  });

    // Inject the link attributes so we get the right package name and version
    if attr::find_linkage_metas(crate.attrs).is_empty() {
        let short_name_to_use = match what {
            Test  => fmt!("%stest", pkg_id.short_name),
            Bench => fmt!("%sbench", pkg_id.short_name),
            _     => pkg_id.short_name.clone()
        };
        debug!("Injecting link name: %s", short_name_to_use);
        let link_options =
            ~[attr::mk_name_value_item_str(@"name", short_name_to_use.to_managed()),
              attr::mk_name_value_item_str(@"vers", pkg_id.version.to_str().to_managed())];

        crate = @ast::Crate {
            attrs: ~[attr::mk_attr(attr::mk_list_item(@"link", link_options))],
            .. (*crate).clone()
        };
    }

    debug!("calling compile_crate_from_input, out_dir = %s,
           building_library = %?", out_dir.to_str(), sess.building_library);
    compile_crate_from_input(&input, out_dir, sess, crate, cfg.clone(), driver::cu_expand);
    true
}

// Should use workcache to avoid recompiling when not necessary
// Should also rename this to something better
// If crate_opt is present, then finish compilation. If it's None, then
// call compile_upto and return the crate
// also, too many arguments
pub fn compile_crate_from_input(input: &driver::input,
                                build_dir: &Path,
                                sess: session::Session,
                                crate: @ast::Crate,
                                cfg: ast::CrateConfig,
                                compile_from: driver::compile_phase) {
    debug!("Calling build_output_filenames with %s, building library? %?",
           build_dir.to_str(), sess.building_library);

    // bad copy
    let outputs = driver::build_output_filenames(input, &Some((*build_dir).clone()), &None,
                                                 crate.attrs, sess);

    debug!("Outputs are %? and output type = %?", outputs, sess.opts.output_type);
    debug!("additional libraries:");
    for sess.opts.addl_lib_search_paths.iter().advance |lib| {
        debug!("an additional library: %s", lib.to_str());
    }

    driver::compile_rest(sess,
                         cfg,
                         compile_upto {
                             from: compile_from,
                             to: driver::cu_everything
                         },
                         Some(outputs),
                         Some(crate));
}

#[cfg(windows)]
pub fn exe_suffix() -> ~str { ~".exe" }

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn exe_suffix() -> ~str { ~"" }

// Called by build_crates
// FIXME (#4432): Use workcache to only compile when needed
pub fn compile_crate(ctxt: &Ctx, pkg_id: &PkgId,
                     crate: &Path, dir: &Path,
                     flags: &[~str], cfgs: &[~str], opt: bool,
                     what: OutputType) -> bool {
    debug!("compile_crate: crate=%s, dir=%s", crate.to_str(), dir.to_str());
    debug!("compile_crate: short_name = %s, flags =...", pkg_id.to_str());
    for flags.iter().advance |fl| {
        debug!("+++ %s", *fl);
    }
    compile_input(ctxt, pkg_id, crate, dir, flags, cfgs, opt, what)
}


/// Collect all `extern mod` directives in `c`, then
/// try to install their targets, failing if any target
/// can't be found.
pub fn find_and_install_dependencies(ctxt: &Ctx,
                                 sess: session::Session,
                                 workspace: &Path,
                                 c: &ast::Crate,
                                 save: @fn(Path)
                                ) {
    // :-(
    debug!("In find_and_install_dependencies...");
    let my_workspace = (*workspace).clone();
    let my_ctxt      = *ctxt;
    for c.each_view_item() |vi: &ast::view_item| {
        debug!("A view item!");
        match vi.node {
            // ignore metadata, I guess
            ast::view_item_extern_mod(lib_ident, _, _) => {
                match my_ctxt.sysroot_opt {
                    Some(ref x) => debug!("sysroot: %s", x.to_str()),
                    None => debug!("No sysroot given")
                };
                let lib_name = sess.str_of(lib_ident);
                match find_library_in_search_path(my_ctxt.sysroot_opt, lib_name) {
                    Some(installed_path) => {
                        debug!("It exists: %s", installed_path.to_str());
                    }
                    None => {
                        // Try to install it
                        let pkg_id = PkgId::new(lib_name, &os::getcwd());
                        my_ctxt.install(&my_workspace, &pkg_id);
                        // Also, add an additional search path
                        debug!("let installed_path...")
                        let installed_path = target_library_in_workspace(&pkg_id,
                                                                         &my_workspace).pop();
                        debug!("Great, I installed %s, and it's in %s",
                               lib_name, installed_path.to_str());
                        save(installed_path);
                    }
                }
            }
            // Ignore `use`s
            _ => ()
        }
    }
}

#[cfg(windows)]
pub fn link_exe(_src: &Path, _dest: &Path) -> bool {
    /* FIXME (#1768): Investigate how to do this on win32
       Node wraps symlinks by having a .bat,
       but that won't work with minGW. */

    false
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn link_exe(src: &Path, dest: &Path) -> bool {
    use std::libc;
    unsafe {
        do src.to_str().as_c_str |src_buf| {
            do dest.to_str().as_c_str |dest_buf| {
                libc::link(src_buf, dest_buf) == 0 as libc::c_int &&
                    libc::chmod(dest_buf, 755) == 0 as libc::c_int
            }
        }
    }
}

pub fn mk_string_lit(s: @str) -> ast::lit {
    spanned {
        node: ast::lit_str(s),
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

// tjc: cheesy
fn debug_flags() -> ~[~str] { ~[] }
// static DEBUG_FLAGS: ~[~str] = ~[~"-Z", ~"time-passes"];
