// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;
use core::{io, libc, os, result, str};
use rustc::driver::{driver, session};
use rustc::metadata::filesearch;
use extra::getopts::groups::getopts;
use extra::term;
use syntax::ast_util::*;
use syntax::codemap::{dummy_sp, spanned};
use syntax::codemap::dummy_spanned;
use syntax::ext::base::ExtCtxt;
use syntax::{ast, attr, codemap, diagnostic, fold};
use syntax::ast::{meta_name_value, meta_list};
use syntax::attr::{mk_attr};
use rustc::back::link::output_type_exe;
use rustc::driver::driver::compile_upto;
use rustc::driver::session::{lib_crate, bin_crate};
use context::Ctx;
use package_id::PkgId;
use path_util::target_library_in_workspace;
use search::find_library_in_search_path;
pub use target::{OutputType, Main, Lib, Bench, Test};

static Commands: &'static [&'static str] =
    &["build", "clean", "do", "info", "install", "prefer", "test", "uninstall",
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
    Commands.any(|&c| c == cmd)
}

struct ListenerFn {
    cmds: ~[~str],
    span: codemap::span,
    path: ~[ast::ident]
}

struct ReadyCtx {
    sess: session::Session,
    crate: @ast::crate,
    ext_cx: @ExtCtxt,
    path: ~[ast::ident],
    fns: ~[ListenerFn]
}

fn fold_mod(_ctx: @mut ReadyCtx,
            m: &ast::_mod,
            fold: @fold::ast_fold) -> ast::_mod {
    fn strip_main(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: do item.attrs.filtered |attr| {
                *attr::get_attr_name(attr) != ~"main"
            },
            .. copy *item
        }
    }

    fold::noop_fold_mod(&ast::_mod {
        items: do m.items.map |item| {
            strip_main(*item)
        },
        .. copy *m
    }, fold)
}

fn fold_item(ctx: @mut ReadyCtx,
             item: @ast::item,
             fold: @fold::ast_fold) -> Option<@ast::item> {
    ctx.path.push(item.ident);

    let attrs = attr::find_attrs_by_name(item.attrs, "pkg_do");

    if attrs.len() > 0 {
        let mut cmds = ~[];

        for attrs.each |attr| {
            match attr.node.value.node {
                ast::meta_list(_, ref mis) => {
                    for mis.each |mi| {
                        match mi.node {
                            ast::meta_word(cmd) => cmds.push(copy *cmd),
                            _ => {}
                        };
                    }
                }
                _ => cmds.push(~"build")
            };
        }

        ctx.fns.push(ListenerFn {
            cmds: cmds,
            span: item.span,
            path: /*bad*/copy ctx.path
        });
    }

    let res = fold::noop_fold_item(item, fold);

    ctx.path.pop();

    res
}

/// Generate/filter main function, add the list of commands, etc.
pub fn ready_crate(sess: session::Session,
                   crate: @ast::crate) -> @ast::crate {
    let ctx = @mut ReadyCtx {
        sess: sess,
        crate: crate,
        ext_cx: ExtCtxt::new(sess.parse_sess, copy sess.opts.cfg),
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

pub fn need_dir(s: &Path) {
    if !os::path_is_dir(s) && !os::make_dir(s, 493_i32) {
        fail!("can't create dir: %s", s.to_str());
    }
}

fn pretty_message<'a>(msg: &'a str, prefix: &'a str, color: u8, out: @io::Writer) {
    let term = term::Terminal::new(out);
    match term {
        Ok(ref t) => {
            t.fg(color);
            out.write_str(prefix);
            t.reset();
        },
        _ => {
            out.write_str(prefix);
        }
    }
    out.write_line(msg);
}

pub fn note(msg: &str) {
    pretty_message(msg, "note: ", term::color_green, io::stdout())
}

pub fn warn(msg: &str) {
    pretty_message(msg, "warning: ", term::color_yellow, io::stdout())
}

pub fn error(msg: &str) {
    pretty_message(msg, "error: ", term::color_red, io::stdout())
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
    let input = driver::file_input(copy *in_file);
    debug!("compile_input: %s / %?", in_file.to_str(), what);
    // tjc: by default, use the package ID name as the link name
    // not sure if we should support anything else

    let binary = @(copy os::args()[0]);

    debug!("flags: %s", str::connect(flags, " "));
    debug!("cfgs: %s", str::connect(cfgs, " "));
    debug!("compile_input's sysroot = %?", ctxt.sysroot_opt);

    let crate_type = match what {
        Lib => lib_crate,
        Test | Bench | Main => bin_crate
    };
    let matches = getopts(~[~"-Z", ~"time-passes"]
                          + match what {
                              Lib => ~[~"--lib"],
                              // --test compiles both #[test] and #[bench] fns
                              Test | Bench => ~[~"--test"],
                              Main => ~[]
                          }
                          + flags
                          + cfgs.flat_map(|&c| { ~[~"--cfg", c] }),
                          driver::optgroups()).get();
    let options = @session::options {
        crate_type: crate_type,
        optimize: if opt { session::Aggressive } else { session::No },
        test: what == Test || what == Bench,
        maybe_sysroot: ctxt.sysroot_opt,
        addl_lib_search_paths: @mut ~[copy *out_dir],
        // output_type should be conditional
        output_type: output_type_exe, // Use this to get a library? That's weird
        .. copy *driver::build_session_options(binary, &matches, diagnostic::emit)
    };

    let addl_lib_search_paths = @mut options.addl_lib_search_paths;

    let sess = driver::build_session(options, diagnostic::emit);

    // Infer dependencies that rustpkg needs to build, by scanning for
    // `extern mod` directives.
    let cfg = driver::build_configuration(sess, binary, &input);
    let (crate_opt, _) = driver::compile_upto(sess, copy cfg, &input, driver::cu_expand, None);

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
    if attr::find_linkage_metas(crate.node.attrs).is_empty() {
        let short_name_to_use = match what {
            Test  => fmt!("%stest", pkg_id.short_name),
            Bench => fmt!("%sbench", pkg_id.short_name),
            _     => copy pkg_id.short_name
        };
        debug!("Injecting link name: %s", short_name_to_use);
        crate = @codemap::respan(crate.span, ast::crate_ {
            attrs: ~[mk_attr(@dummy_spanned(
                meta_list(@~"link",
                 ~[@dummy_spanned(meta_name_value(@~"name",
                                      mk_string_lit(@short_name_to_use))),
                   @dummy_spanned(meta_name_value(@~"vers",
                                      mk_string_lit(@(copy pkg_id.version.to_str()))))])))],
            ..copy crate.node});
    }

    debug!("calling compile_crate_from_input, out_dir = %s,
           building_library = %?", out_dir.to_str(), sess.building_library);
    compile_crate_from_input(&input, out_dir, sess, crate, copy cfg);
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
                                crate: @ast::crate,
                                cfg: ast::crate_cfg) {
    debug!("Calling build_output_filenames with %s, building library? %?",
           build_dir.to_str(), sess.building_library);

    // bad copy
    let outputs = driver::build_output_filenames(input, &Some(copy *build_dir), &None,
                                                 crate.node.attrs, sess);

    debug!("Outputs are %? and output type = %?", outputs, sess.opts.output_type);
    debug!("additional libraries:");
    for sess.opts.addl_lib_search_paths.each |lib| {
        debug!("an additional library: %s", lib.to_str());
    }

    driver::compile_rest(sess,
                         cfg,
                         compile_upto {
                             from: driver::cu_expand,
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
    for flags.each |&fl| {
        debug!("+++ %s", fl);
    }
    compile_input(ctxt, pkg_id, crate, dir, flags, cfgs, opt, what)
}

/// Collect all `extern mod` directives in `c`, then
/// try to install their targets, failing if any target
/// can't be found.
fn find_and_install_dependencies(ctxt: &Ctx,
                                 sess: session::Session,
                                 workspace: &Path,
                                 c: &ast::crate,
                                 save: @fn(Path)
                                ) {
    // :-(
    debug!("In find_and_install_dependencies...");
    let my_workspace = copy *workspace;
    let my_ctxt      = copy *ctxt;
    for c.each_view_item() |vi: @ast::view_item| {
        debug!("A view item!");
        match vi.node {
            // ignore metadata, I guess
            ast::view_item_extern_mod(lib_ident, _, _) => {
                match my_ctxt.sysroot_opt {
                    Some(ref x) => debug!("sysroot: %s", x.to_str()),
                    None => ()
                };
                let lib_name = sess.str_of(lib_ident);
                match find_library_in_search_path(my_ctxt.sysroot_opt, *lib_name) {
                    Some(installed_path) => {
                        debug!("It exists: %s", installed_path.to_str());
                    }
                    None => {
                        // Try to install it
                        let pkg_id = PkgId::new(*lib_name);
                        my_ctxt.install(&my_workspace, &pkg_id);
                        // Also, add an additional search path
                        let installed_path = target_library_in_workspace(&pkg_id,
                                                                         &my_workspace).pop();
                        debug!("Great, I installed %s, and it's in %s",
                               *lib_name, installed_path.to_str());
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
    unsafe {
        do str::as_c_str(src.to_str()) |src_buf| {
            do str::as_c_str(dest.to_str()) |dest_buf| {
                libc::link(src_buf, dest_buf) == 0 as libc::c_int &&
                    libc::chmod(dest_buf, 755) == 0 as libc::c_int
            }
        }
    }
}

pub fn mk_string_lit(s: @~str) -> ast::lit {
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
