// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::*;
use core::cmp::Ord;
use core::hash::Streaming;
use core::rt::io::Writer;
use rustc::driver::{driver, session};
use rustc::driver::session::{lib_crate, unknown_crate};
use rustc::metadata::filesearch;
use std::getopts::groups::getopts;
use std::semver;
use std::{json, term, getopts};
use syntax::ast_util::*;
use syntax::codemap::{dummy_sp, spanned, dummy_spanned};
use syntax::ext::base::{mk_ctxt, ext_ctxt};
use syntax::ext::build;
use syntax::{ast, attr, codemap, diagnostic, fold};
use syntax::ast::{meta_name_value, meta_list, attribute, crate_};
use syntax::attr::{mk_attr};
use rustc::back::link::output_type_exe;
use rustc::driver::session::{lib_crate, unknown_crate, crate_type};

pub type ExitCode = int; // For now

/// A version is either an exact revision,
/// or a semantic version
pub enum Version {
    ExactRevision(float),
    SemVersion(semver::Version)
}

impl Ord for Version {
    fn lt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 < f2,
            (&SemVersion(v1), &SemVersion(v2)) => v1 < v2,
            _ => false // incomparable, really
        }
    }
    fn le(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 <= f2,
            (&SemVersion(v1), &SemVersion(v2)) => v1 <= v2,
            _ => false // incomparable, really
        }
    }
    fn ge(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 > f2,
            (&SemVersion(v1), &SemVersion(v2)) => v1 > v2,
            _ => false // incomparable, really
        }
    }
    fn gt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 >= f2,
            (&SemVersion(v1), &SemVersion(v2)) => v1 >= v2,
            _ => false // incomparable, really
        }
    }

}

impl ToStr for Version {
    fn to_str(&self) -> ~str {
        match *self {
            ExactRevision(n) => n.to_str(),
            SemVersion(v) => v.to_str()
        }
    }
}

/// Placeholder
pub fn default_version() -> Version { ExactRevision(0.1) }

// Path-fragment identifier of a package such as
// 'github.com/graydon/test'; path must be a relative
// path with >=1 component.
pub struct PkgId {
    path: Path,
    version: Version
}

pub impl PkgId {
    fn new(s: &str) -> PkgId {
        use bad_pkg_id::cond;

        let p = Path(s);
        if p.is_absolute {
            return cond.raise((p, ~"absolute pkgid"));
        }
        if p.components.len() < 1 {
            return cond.raise((p, ~"0-length pkgid"));
        }
        PkgId {
            path: p,
            version: default_version()
        }
    }

    fn hash(&self) -> ~str {
        fmt!("%s-%s-%s", self.path.to_str(),
             hash(self.path.to_str() + self.version.to_str()),
             self.version.to_str())
    }

}

impl ToStr for PkgId {
    fn to_str(&self) -> ~str {
        // should probably use the filestem and not the whole path
        fmt!("%s-%s", self.path.to_str(),
             // Replace dots with -s in the version
             // this is because otherwise rustc will think
             // that foo-0.1 has .1 as its extension
             // (Temporary hack until I figure out how to
             // get rustc to not name the object file
             // foo-0.o if I pass in foo-0.1 to build_output_filenames)
             str::replace(self.version.to_str(), ".", "-"))
    }
}

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

pub fn is_cmd(cmd: ~str) -> bool {
    let cmds = &[~"build", ~"clean", ~"do", ~"info", ~"install", ~"prefer",
                 ~"test", ~"uninstall", ~"unprefer"];

    vec::contains(cmds, &cmd)
}

pub fn parse_name(id: ~str) -> result::Result<~str, ~str> {
    let mut last_part = None;

    for str::each_split_char(id, '.') |part| {
        for str::each_char(part) |char| {
            if char::is_whitespace(char) {
                return result::Err(
                    ~"could not parse id: contains whitespace");
            } else if char::is_uppercase(char) {
                return result::Err(
                    ~"could not parse id: should be all lowercase");
            }
        }
        last_part = Some(part.to_owned());
    }
    if last_part.is_none() { return result::Err(~"could not parse id: is empty"); }

    result::Ok(last_part.unwrap())
}

struct ListenerFn {
    cmds: ~[~str],
    span: codemap::span,
    path: ~[ast::ident]
}

struct ReadyCtx {
    sess: session::Session,
    crate: @ast::crate,
    ext_cx: @ext_ctxt,
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

    let attrs = attr::find_attrs_by_name(item.attrs, ~"pkg_do");

    if attrs.len() > 0 {
        let mut cmds = ~[];

        for attrs.each |attr| {
            match attr.node.value.node {
                ast::meta_list(_, mis) => {
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

fn add_pkg_module(ctx: @mut ReadyCtx, m: ast::_mod) -> ast::_mod {
    let listeners = mk_listener_vec(ctx);
    let ext_cx = ctx.ext_cx;
    let item = quote_item! (
        mod __pkg {
            extern mod rustpkg (vers="0.7-pre");
            static listeners : &[rustpkg::Listener] = $listeners;
            #[main]
            fn main() {
                rustpkg::run(listeners);
            }
        }
    );
    ast::_mod {
        items: vec::append_one(/*bad*/copy m.items, item.get()),
        .. m
    }
}

fn mk_listener_vec(ctx: @mut ReadyCtx) -> @ast::expr {
    let fns = ctx.fns;
    let descs = do fns.map |listener| {
        mk_listener_rec(ctx, *listener)
    };
    let ext_cx = ctx.ext_cx;
    build::mk_slice_vec_e(ext_cx, dummy_sp(), descs)
}

fn mk_listener_rec(ctx: @mut ReadyCtx, listener: ListenerFn) -> @ast::expr {
    let span = listener.span;
    let cmds = do listener.cmds.map |&cmd| {
        let ext_cx = ctx.ext_cx;
        build::mk_base_str(ext_cx, span, cmd)
    };

    let ext_cx = ctx.ext_cx;
    let cmds_expr = build::mk_slice_vec_e(ext_cx, span, cmds);
    let cb_expr = build::mk_path(ext_cx, span, copy listener.path);

    quote_expr!(
        Listener {
            cmds: $cmds_expr,
            cb: $cb_expr
        }
    )
}

/// Generate/filter main function, add the list of commands, etc.
pub fn ready_crate(sess: session::Session,
                   crate: @ast::crate) -> @ast::crate {
    let ctx = @mut ReadyCtx {
        sess: sess,
        crate: crate,
        ext_cx: mk_ctxt(sess.parse_sess, copy sess.opts.cfg),
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

pub fn parse_vers(vers: ~str) -> result::Result<semver::Version, ~str> {
    match semver::parse(vers) {
        Some(vers) => result::Ok(vers),
        None => result::Err(~"could not parse version: invalid")
    }
}

pub fn need_dir(s: &Path) {
    if !os::path_is_dir(s) && !os::make_dir(s, 493_i32) {
        fail!("can't create dir: %s", s.to_str());
    }
}

pub fn note(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_green);
        out.write_str(~"note: ");
        term::reset(out);
        out.write_line(msg);
    } else {
        out.write_line(~"note: " + msg);
    }
}

pub fn warn(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_yellow);
        out.write_str(~"warning: ");
        term::reset(out);
        out.write_line(msg);
    } else {
        out.write_line(~"warning: " + msg);
    }
}

pub fn error(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_red);
        out.write_str(~"error: ");
        term::reset(out);
        out.write_line(msg);
    } else {
        out.write_line(~"error: " + msg);
    }
}

pub fn hash(data: ~str) -> ~str {
    let mut hasher = hash::default_state();
    let buffer = str::as_bytes_slice(data);
    hasher.write(buffer);
    hasher.result_str()
}

pub fn temp_change_dir<T>(dir: &Path, cb: &fn() -> T) {
    let cwd = os::getcwd();

    os::change_dir(dir);
    cb();
    os::change_dir(&cwd);
}

pub fn touch(path: &Path) {
    match io::mk_file_writer(path, ~[io::Create]) {
        result::Ok(writer) => writer.write_line(~""),
        _ => {}
    }
}

pub fn remove_dir_r(path: &Path) {
    for os::walk_dir(path) |&file| {
        let mut cdir = file;

        loop {
            if os::path_is_dir(&cdir) {
                os::remove_dir(&cdir);
            } else {
                os::remove_file(&cdir);
            }

            cdir = cdir.dir_path();

            if cdir == *path { break; }
        }
    }

    os::remove_dir(path);
}

pub fn wait_for_lock(path: &Path) {
    if os::path_exists(path) {
        warn(fmt!("the database appears locked, please wait (or rm %s)",
                        path.to_str()));

        loop {
            if !os::path_exists(path) { break; }
        }
    }
}

pub fn load_pkgs() -> result::Result<~[json::Json], ~str> {
    fail!("load_pkg not implemented");
}

pub fn get_pkg(_id: ~str,
               _vers: Option<~str>) -> result::Result<Pkg, ~str> {
    fail!("get_pkg not implemented");
}

pub fn add_pkg(pkg: &Pkg) -> bool {
    note(fmt!("Would be adding package, but add_pkg is not yet implemented %s",
         pkg.to_str()));
    false
}

// FIXME (#4432): Use workcache to only compile when needed
pub fn compile_input(sysroot: Option<@Path>,
                     pkg_id: PkgId,
                     in_file: &Path,
                     out_dir: &Path,
                     flags: ~[~str],
                     cfgs: ~[~str],
                     opt: bool,
                     test: bool,
                     crate_type: session::crate_type) -> bool {

    // Want just the directory component here
    let pkg_filename = pkg_id.path.filename().expect(~"Weird pkg id");
    let short_name = fmt!("%s-%s", pkg_filename, pkg_id.version.to_str());

    assert!(in_file.components.len() > 1);
    let input = driver::file_input(copy *in_file);
    debug!("compile_input: %s / %?", in_file.to_str(), crate_type);
    // tjc: by default, use the package ID name as the link name
    // not sure if we should support anything else

    let binary = os::args()[0];
    let building_library = match crate_type {
        lib_crate | unknown_crate => true,
        _ => false
    };

    let out_file = if building_library {
        out_dir.push(os::dll_filename(short_name))
    }
    else {
        out_dir.push(short_name + if test { ~"test" } else { ~"" }
                     + os::EXE_SUFFIX)
    };

    debug!("compiling %s into %s",
           in_file.to_str(),
           out_file.to_str());
    debug!("flags: %s", str::connect(flags, ~" "));
    debug!("cfgs: %s", str::connect(cfgs, ~" "));
    debug!("compile_input's sysroot = %?", sysroot);

    let matches = getopts(~[~"-Z", ~"time-passes"]
                          + if building_library { ~[~"--lib"] }
                            else if test { ~[~"--test"] }
                            // bench?
                            else { ~[] }
                          + flags
                          + cfgs.flat_map(|&c| { ~[~"--cfg", c] }),
                          driver::optgroups()).get();
    let options = @session::options {
        crate_type: crate_type,
        optimize: if opt { session::Aggressive } else { session::No },
        test: test,
        maybe_sysroot: sysroot,
        addl_lib_search_paths: ~[copy *out_dir],
        .. *driver::build_session_options(@binary, &matches, diagnostic::emit)
    };
    let mut crate_cfg = options.cfg;

    for cfgs.each |&cfg| {
        crate_cfg.push(attr::mk_word_item(@cfg));
    }

    let options = @session::options {
        cfg: vec::append(options.cfg, crate_cfg),
        // output_type should be conditional
        output_type: output_type_exe, // Use this to get a library? That's weird
        .. *options
    };
    let sess = driver::build_session(options, diagnostic::emit);

    debug!("calling compile_crate_from_input, out_dir = %s,
           building_library = %?", out_dir.to_str(), sess.building_library);
    let _ = compile_crate_from_input(input, pkg_id, Some(*out_dir), sess, None,
                                     out_file, binary,
                                     driver::cu_everything);
    true
}

// Should use workcache to avoid recompiling when not necessary
// Should also rename this to something better
// If crate_opt is present, then finish compilation. If it's None, then
// call compile_upto and return the crate
// also, too many arguments
pub fn compile_crate_from_input(input: driver::input,
                                pkg_id: PkgId,
                                build_dir_opt: Option<Path>,
                                sess: session::Session,
                                crate_opt: Option<@ast::crate>,
                                out_file: Path,
                                binary: ~str,
                                what: driver::compile_upto) -> @ast::crate {
    debug!("Calling build_output_filenames with %? and %s", build_dir_opt, out_file.to_str());
    let outputs = driver::build_output_filenames(&input, &build_dir_opt, &Some(out_file), sess);
    debug!("Outputs are %? and output type = %?", outputs, sess.opts.output_type);
    let cfg = driver::build_configuration(sess, @binary, &input);
    match crate_opt {
        Some(c) => {
            debug!("Calling compile_rest, outputs = %?", outputs);
            assert!(what == driver::cu_everything);
            driver::compile_rest(sess, cfg, driver::cu_everything, Some(outputs), Some(c));
            c
        }
        None => {
            debug!("Calling compile_upto, outputs = %?", outputs);
            let (crate, _) = driver::compile_upto(sess, cfg, &input,
                                                  driver::cu_parse, Some(outputs));

            debug!("About to inject link_meta info...");
            // Inject the inferred link_meta info if it's not already there
            // (assumes that name and vers are the only linkage metas)
            let mut crate_to_use = crate;

            debug!("How many attrs? %?", attr::find_linkage_metas(crate.node.attrs).len());

            if attr::find_linkage_metas(crate.node.attrs).is_empty() {
                crate_to_use = add_attrs(*crate, ~[mk_attr(@dummy_spanned(meta_list(@~"link",
                                                  // change PkgId to have a <shortname> field?
                    ~[@dummy_spanned(meta_name_value(@~"name",
                                                    mk_string_lit(@pkg_id.path.filestem().get()))),
                      @dummy_spanned(meta_name_value(@~"vers",
                                                    mk_string_lit(@pkg_id.version.to_str())))])))]);
            }

            driver::compile_rest(sess, cfg, what, Some(outputs), Some(crate_to_use));
            crate_to_use
        }
    }
}

#[cfg(windows)]
pub fn exe_suffix() -> ~str { ~".exe" }

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn exe_suffix() -> ~str { ~"" }


/// Returns a copy of crate `c` with attributes `attrs` added to its
/// attributes
fn add_attrs(c: ast::crate, new_attrs: ~[attribute]) -> @ast::crate {
    @spanned {
        node: crate_ {
            attrs: c.node.attrs + new_attrs, ..c.node
        },
        span: c.span
    }
}

// Called by build_crates
// FIXME (#4432): Use workcache to only compile when needed
pub fn compile_crate(sysroot: Option<@Path>, pkg_id: PkgId,
                     crate: &Path, dir: &Path,
                     flags: ~[~str], cfgs: ~[~str], opt: bool,
                     test: bool, crate_type: crate_type) -> bool {
    debug!("compile_crate: crate=%s, dir=%s", crate.to_str(), dir.to_str());
    debug!("compile_crate: short_name = %s, flags =...", pkg_id.to_str());
    for flags.each |&fl| {
        debug!("+++ %s", fl);
    }
    compile_input(sysroot, pkg_id,
                  crate, dir, flags, cfgs, opt, test, crate_type)
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
    use super::{is_cmd, parse_name};

    #[test]
    fn test_is_cmd() {
        assert!(is_cmd(~"build"));
        assert!(is_cmd(~"clean"));
        assert!(is_cmd(~"do"));
        assert!(is_cmd(~"info"));
        assert!(is_cmd(~"install"));
        assert!(is_cmd(~"prefer"));
        assert!(is_cmd(~"test"));
        assert!(is_cmd(~"uninstall"));
        assert!(is_cmd(~"unprefer"));
    }

    #[test]
    fn test_parse_name() {
        assert!(parse_name(~"org.mozilla.servo").get() == ~"servo");
        assert!(parse_name(~"org. mozilla.servo 2131").is_err());
    }
}
