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
use send_map::linear::LinearMap;
use rustc::metadata::filesearch;
use rustc::driver::{driver, session};
use syntax::ast_util::*;
use syntax::{ast, attr, codemap, diagnostic, fold, parse, visit};
use codemap::span;
use semver::Version;
use std::{json, term, sort};
use api::Listener;

pub struct Package {
    id: ~str,
    vers: Version,
    bins: ~[~str],
    libs: ~[~str],
}

pub fn root() -> Path {
    match filesearch::get_rustpkg_root() {
        result::Ok(path) => path,
        result::Err(err) => fail err
    }
}

pub fn is_cmd(cmd: ~str) -> bool {
    let cmds = &[~"build", ~"clean", ~"do", ~"install", ~"prefer",
                 ~"test", ~"uninstall", ~"unprefer"];

    vec::contains(cmds, &cmd)
}

pub fn parse_name(id: ~str) -> result::Result<~str, ~str> {
    let parts = str::split_char(id, '.');

    for parts.each |&part| {
        for str::chars(part).each |&char| {
            if char::is_whitespace(char) {
                return result::Err(~"could not parse id: contains whitespace");
            } else if char::is_uppercase(char) {
                return result::Err(~"could not parse id: should be all lowercase");
            }
        }
    }

    result::Ok(parts.last())
}

fn mk_rustpkg_use(ctx: @ReadyCtx) -> @ast::view_item {
    let vers = ast::lit_str(@~"0.6");
    let vers = no_span(vers);
    let mi = ast::meta_name_value(~"vers", vers);
    let mi = no_span(mi);
    let vi = ast::view_item_use(ctx.sess.ident_of(~"rustpkg"),
                                ~[@mi],
                                ctx.sess.next_node_id());

    @ast::view_item {
        node: vi,
        attrs: ~[],
        vis: ast::private,
        span: dummy_sp()
    }
}

struct ListenerFn {
    cmds: ~[~str],
    span: codemap::span,
    path: ~[ast::ident]
}

struct ReadyCtx {
    sess: session::Session,
    crate: @ast::crate,
    mut path: ~[ast::ident],
    mut fns: ~[ListenerFn]
}

fn fold_mod(ctx: @ReadyCtx, m: ast::_mod,
            fold: fold::ast_fold) -> ast::_mod {
    fn strip_main(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: do item.attrs.filtered |attr| {
                attr::get_attr_name(*attr) != ~"main"
            },
            .. copy *item
        }
    }

    fold::noop_fold_mod({
        view_items: vec::append_one(m.view_items, mk_rustpkg_use(ctx)),
        items: do vec::map(m.items) |item| {
            strip_main(*item)
        }
    }, fold)
}

fn fold_crate(ctx: @ReadyCtx, crate: ast::crate_,
              fold: fold::ast_fold) -> ast::crate_ {
    let folded = fold::noop_fold_crate(crate, fold);

    ast::crate_ {
        module: add_pkg_module(ctx, /*bad*/copy folded.module),
        .. folded
    }
}

fn fold_item(ctx: @ReadyCtx, &&item: @ast::item,
             fold: fold::ast_fold) -> Option<@ast::item> {

    ctx.path.push(item.ident);

    let attrs = attr::find_attrs_by_name(item.attrs, ~"pkg_do");

    if attrs.len() > 0 {
        let mut cmds = ~[];

        for attrs.each |attr| {
            match attr.node.value.node {
                ast::meta_list(_, mis) => {
                    for mis.each |mi| {
                        match mi.node {
                            ast::meta_word(cmd) => cmds.push(cmd),
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

fn mk_rustpkg_import(ctx: @ReadyCtx) -> @ast::view_item {
    let path = @ast::path {
        span: dummy_sp(),
        global: false,
        idents: ~[ctx.sess.ident_of(~"rustpkg")],
        rp: None,
        types: ~[]
    };
    let vp = @no_span(ast::view_path_simple(ctx.sess.ident_of(~"rustpkg"),
                                            path, ast::type_value_ns,
                                            ctx.sess.next_node_id()));

    @ast::view_item {
        node: ast::view_item_import(~[vp]),
        attrs: ~[],
        vis: ast::private,
        span: dummy_sp()
    }
}

fn add_pkg_module(ctx: @ReadyCtx, m: ast::_mod) -> ast::_mod {
    let listeners = mk_listeners(ctx);
    let main = mk_main(ctx);
    let pkg_mod = @{
        view_items: ~[mk_rustpkg_import(ctx)],
        items: ~[main, listeners]
    };
    let resolve_unexported_attr =
        attr::mk_attr(attr::mk_word_item(~"!resolve_unexported"));
    let item_ = ast::item_mod(*pkg_mod);
    let item = @ast::item {
        ident: ctx.sess.ident_of(~"__pkg"),
        attrs: ~[resolve_unexported_attr],
        id: ctx.sess.next_node_id(),
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
    };

    {
        items: vec::append_one(/*bad*/copy m.items, item),
        .. m
    }
}

fn no_span<T: Copy>(t: T) -> ast::spanned<T> {
    ast::spanned {
        node: t,
        span: dummy_sp()
    }
}

fn path_node(ids: ~[ast::ident]) -> @ast::path {
    @ast::path {
        span: dummy_sp(),
        global: false,
        idents: ids,
        rp: None,
        types: ~[]
    }
}

fn path_node_global(ids: ~[ast::ident]) -> @ast::path {
    @ast::path {
        span: dummy_sp(),
        global: true,
        idents: ids,
        rp: None,
        types: ~[]
    }
}

fn mk_listeners(ctx: @ReadyCtx) -> @ast::item {
    let ret_ty = mk_listener_vec_ty(ctx);
    let decl = {
        inputs: ~[],
        output: ret_ty,
        cf: ast::return_val
    };
    let listeners = mk_listener_vec(ctx);
    let body_ = default_block(~[], option::Some(listeners),
                              ctx.sess.next_node_id());
    let body = no_span(body_);
    let item_ = ast::item_fn(decl, ast::impure_fn, ~[], body);

    @ast::item {
        ident: ctx.sess.ident_of(~"listeners"),
        attrs: ~[],
        id: ctx.sess.next_node_id(),
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
    }
}

fn mk_path(ctx: @ReadyCtx, path: ~[ast::ident]) -> @ast::path {
    path_node(~[ctx.sess.ident_of(~"rustpkg")] + path)
}

fn mk_listener_vec_ty(ctx: @ReadyCtx) -> @ast::Ty {
    let listener_ty_path = mk_path(ctx, ~[ctx.sess.ident_of(~"Listener")]);
    let listener_ty = {
        id: ctx.sess.next_node_id(),
        node: ast::ty_path(listener_ty_path,
                           ctx.sess.next_node_id()),
        span: dummy_sp()
    };
    let vec_mt = ast::mt {
        ty: @listener_ty,
        mutbl: ast::m_imm
    };
    let inner_ty = @{
        id: ctx.sess.next_node_id(),
        node: ast::ty_vec(vec_mt),
        span: dummy_sp()
    };

    @{
        id: ctx.sess.next_node_id(),
        node: ast::ty_uniq(ast::mt {
            ty: inner_ty,
            mutbl: ast::m_imm
        }),
        span: dummy_sp()
    }
}

fn mk_listener_vec(ctx: @ReadyCtx) -> @ast::expr {
    let fns = ctx.fns;

    let descs = do fns.map |listener| {
        mk_listener_rec(ctx, *listener)
    };
    let inner_expr = @{
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_vec(descs, ast::m_imm),
        span: dummy_sp()
    };

    @{
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_vstore(inner_expr, ast::expr_vstore_uniq),
        span: dummy_sp()
    }
}

fn mk_listener_rec(ctx: @ReadyCtx, listener: ListenerFn) -> @ast::expr {
    let span = listener.span;
    let path = /*bad*/copy listener.path;
    let descs = do listener.cmds.map |&cmd| {
        let inner = @{
            id: ctx.sess.next_node_id(),
            callee_id: ctx.sess.next_node_id(),
            node: ast::expr_lit(@no_span(ast::lit_str(@cmd))),
            span: span
        };

        @{
            id: ctx.sess.next_node_id(),
            callee_id: ctx.sess.next_node_id(),
            node: ast::expr_vstore(inner, ast::expr_vstore_uniq),
            span: dummy_sp()
        }
    };
    let cmd_expr_inner = @{
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_vec(descs, ast::m_imm),
        span: dummy_sp()
    };
    let cmd_expr = {
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_vstore(cmd_expr_inner, ast::expr_vstore_uniq),
        span: dummy_sp()
    };
    let cmd_field = no_span(ast::field_ {
        mutbl: ast::m_imm,
        ident: ctx.sess.ident_of(~"cmds"),
        expr: @cmd_expr,
    });

    let cb_path = path_node_global(path);
    let cb_expr = {
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_path(cb_path),
        span: span
    };
    let cb_wrapper_expr = mk_fn_wrapper(ctx, cb_expr, span);
    let cb_field = no_span(ast::field_ {
        mutbl: ast::m_imm,
        ident: ctx.sess.ident_of(~"cb"),
        expr: cb_wrapper_expr
    });

    let listener_path = mk_path(ctx, ~[ctx.sess.ident_of(~"Listener")]);
    let listener_rec_ = ast::expr_struct(listener_path,
                                         ~[cmd_field, cb_field],
                                         option::None);
    @{
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: listener_rec_,
        span: span
    }
}

fn mk_fn_wrapper(ctx: @ReadyCtx, fn_path_expr: ast::expr,
                 span: span) -> @ast::expr {
    let call_expr = {
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_call(@fn_path_expr, ~[], false),
        span: span
    };
    let call_stmt = no_span(ast::stmt_semi(@call_expr, ctx.sess.next_node_id()));
    let wrapper_decl = {
        inputs: ~[],
        output: @{
            id: ctx.sess.next_node_id(),
            node: ast::ty_nil, span: span
        },
        cf: ast::return_val
    };
    let wrapper_body = no_span(ast::blk_ {
        view_items: ~[],
        stmts: ~[@call_stmt],
        expr: option::None,
        id: ctx.sess.next_node_id(),
        rules: ast::default_blk
    });

    @{
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: ast::expr_fn(ast::ProtoBare, wrapper_decl,
                           wrapper_body, @~[]),
        span: span
    }
}

fn mk_main(ctx: @ReadyCtx) -> @ast::item {
    let ret_ty = {
        id: ctx.sess.next_node_id(),
        node: ast::ty_nil,
        span: dummy_sp()
    };
    let decl = {
        inputs: ~[],
        output: @ret_ty,
        cf: ast::return_val
    };
    let run_call_expr = mk_run_call(ctx);
    let body_ = default_block(~[], option::Some(run_call_expr),
                              ctx.sess.next_node_id());
    let body = ast::spanned {
        node: body_,
        span: dummy_sp()
    };
    let item_ = ast::item_fn(decl, ast::impure_fn, ~[], body);

    @ast::item {
        ident: ctx.sess.ident_of(~"main"),
        attrs: ~[attr::mk_attr(attr::mk_word_item(~"main"))],
        id: ctx.sess.next_node_id(),
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
    }
}

fn mk_run_call(ctx: @ReadyCtx) -> @ast::expr {
    let listener_path = path_node(~[ctx.sess.ident_of(~"listeners")]);
    let listener_path_expr_ = ast::expr_path(listener_path);
    let listener_path_expr = {
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: listener_path_expr_,
        span: dummy_sp()
    };
    let listener_call_expr_ = ast::expr_call(@listener_path_expr, ~[], false);
    let listener_call_expr = {
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: listener_call_expr_,
        span: dummy_sp()
    };
    let rustpkg_run_path = mk_path(ctx, ~[ctx.sess.ident_of(~"run")]);

    let rustpkg_run_path_expr_ = ast::expr_path(rustpkg_run_path);
    let rustpkg_run_path_expr = {
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: rustpkg_run_path_expr_,
        span: dummy_sp()
    };
    let rustpkg_run_call_expr_ = ast::expr_call(@rustpkg_run_path_expr,
                                               ~[@listener_call_expr],
                                               false);
    @{
        id: ctx.sess.next_node_id(),
        callee_id: ctx.sess.next_node_id(),
        node: rustpkg_run_call_expr_,
        span: dummy_sp()
    }
}

/// Generate/filter main function, add the list of commands, etc.
pub fn ready_crate(sess: session::Session,
                   crate: @ast::crate) -> @ast::crate {
    let ctx = @ReadyCtx {
        sess: sess,
        crate: crate,
        mut path: ~[],
        mut fns: ~[]
    };
    let precursor = @fold::AstFoldFns {
        fold_crate: fold::wrap(|a, b| fold_crate(ctx, a, b)),
        fold_item: |a, b| fold_item(ctx, a, b),
        fold_mod: |a, b| fold_mod(ctx, a, b),
        .. *fold::default_ast_fold()
    };

    let fold = fold::make_fold(precursor);

    @fold.fold_crate(*crate)
}

pub fn parse_vers(vers: ~str) -> result::Result<Version, ~str> {
    match semver::parse(vers) {
        Some(vers) => result::Ok(vers),
        None => result::Err(~"could not parse version: invalid")
    }
}

pub fn need_dir(s: &Path) {
    if !os::path_is_dir(s) && !os::make_dir(s, 493_i32) {
        fail fmt!("can't create dir: %s", s.to_str());
    }
}

pub fn note(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_green);
        out.write_str(~"note: ");
        term::reset(out);
        out.write_line(msg);
    } else { out.write_line(~"note: " + msg); }
}

pub fn warn(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_yellow);
        out.write_str(~"warning: ");
        term::reset(out);
        out.write_line(msg);
    }else { out.write_line(~"warning: " + msg); }
}

pub fn error(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_red);
        out.write_str(~"error: ");
        term::reset(out);
        out.write_line(msg);
    }
    else { out.write_line(~"error: " + msg); }
}

pub fn hash(data: ~str) -> ~str {
    let hasher = hash::default_state();

    hasher.write_str(data);
    hasher.result_str()
}

pub fn temp_change_dir<T>(dir: &Path, cb: fn() -> T) {
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

fn _add_pkg(packages: ~[json::Json], pkg: &Package) -> ~[json::Json] {
    for packages.each |&package| {
        match package {
            json::Object(map) => {
                let mut has_id = false;

                match map.get(&~"id") {
                    json::String(str) => {
                        if pkg.id == str {
                            has_id = true;
                        }
                    }
                    _ => {}
                }

                match map.get(&~"vers") {
                    json::String(str) => {
                        if pkg.vers.to_str() == str {
                            return packages;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let mut map = ~LinearMap();

    map.insert(~"id", json::String(pkg.id));
    map.insert(~"vers", json::String(pkg.vers.to_str()));
    map.insert(~"bins", json::List(do pkg.bins.map |&bin| {
        json::String(bin)
    }));
    map.insert(~"libs", json::List(do pkg.libs.map |&lib| {
        json::String(lib)
    }));

    vec::append(packages, ~[json::Object(map)])
}

fn _rm_pkg(packages: ~[json::Json], pkg: &Package) -> ~[json::Json] {
    do packages.filter_map |&package| {
        match package {
            json::Object(map) => {
                let mut has_id = false;

                match map.get(&~"id") {
                    json::String(str) => {
                        if pkg.id == str {
                            has_id = true;
                        }
                    }
                    _ => {}
                }

                match map.get(&~"vers") {
                    json::String(str) => {
                        if pkg.vers.to_str() == str { None }
                        else { Some(package) } 
                    }
                    _ => { Some(package) }
                }
            }
            _ => { Some(package) }
        }
    }
}

pub fn load_pkgs() -> result::Result<~[json::Json], ~str> {
    let root = root();
    let db = root.push(~"db.json");
    let db_lock = root.push(~"db.json.lck");

    wait_for_lock(&db_lock);
    touch(&db_lock);

    let packages = if os::path_exists(&db) {
        match io::read_whole_file_str(&db) {
            result::Ok(str) => {
                match json::from_str(str) {
                    result::Ok(json) => {
                        match json {
                            json::List(list) => list,
                            _ => {
                                os::remove_file(&db_lock);

                                return result::Err(~"package db's json is not a list");
                            }
                        }
                    }
                    result::Err(err) => {
                        os::remove_file(&db_lock);

                        return result::Err(fmt!("failed to parse package db: %s", err.to_str()));
                    }
                }
            }
            result::Err(err) => {
                os::remove_file(&db_lock);

                return result::Err(fmt!("failed to read package db: %s", err));
            }
        }
    } else { ~[] };

    os::remove_file(&db_lock);

    result::Ok(packages)
}

pub fn get_pkg(id: ~str, vers: Option<~str>) -> result::Result<Package, ~str> {
    let name = match parse_name(id) {
        result::Ok(name) => name,
        result::Err(err) => return result::Err(err)
    };
    let packages = match load_pkgs() {
        result::Ok(packages) => packages,
        result::Err(err) => return result::Err(err)
    };
    let mut sel = None;
    let mut possibs = ~[];
    let mut err = None;

    for packages.each |&package| {
        match package {
            json::Object(map) => {
                let pid = match map.get(&~"id") {
                    json::String(str) => str,
                    _ => loop
                };
                let pname = match parse_name(pid) {
                    result::Ok(pname) => pname,
                    result::Err(perr) => {
                        err = Some(perr);

                        break;
                    }
                };
                let pvers = match map.get(&~"vers") {
                    json::String(str) => str,
                    _ => loop
                };
                if pid == id || pname == name {
                    let bins = match map.get(&~"bins") {
                        json::List(list) => {
                            do list.map |&bin| {
                                match bin {
                                    json::String(str) => str,
                                    _ => ~""
                                }
                            }
                        }
                        _ => ~[]
                    };
                    let libs = match map.get(&~"libs") {
                        json::List(list) => {
                            do list.map |&lib| {
                                match lib {
                                    json::String(str) => str,
                                    _ => ~""
                                }
                            }
                        }
                        _ => ~[]
                    };
                    let package = Package {
                        id: pid,
                        vers: match parse_vers(pvers) {
                            result::Ok(vers) => vers,
                            result::Err(verr) => {
                                err = Some(verr);

                                break;
                            }
                        },
                        bins: bins,
                        libs: libs
                    };

                    if !vers.is_none() && vers.get() == pvers {
                        sel = Some(package);
                    }
                    else {
                        possibs.push(package);
                    }
                }
            }
            _ => {}
        }
    }

    if !err.is_none() {
        return result::Err(err.get());
    }
    if !sel.is_none() {
        return result::Ok(sel.get());
    }
    if !vers.is_none() || possibs.len() < 1 {
        return result::Err(~"package not found");
    }

    result::Ok(sort::merge_sort(possibs, |v1, v2| {
        v1.vers <= v2.vers
    }).last())
}

pub fn add_pkg(pkg: &Package) -> bool {
    let root = root();
    let db = root.push(~"db.json");
    let db_lock = root.push(~"db.json.lck");
    let packages = match load_pkgs() {
        result::Ok(packages) => packages,
        result::Err(err) => {
            error(err);

            return false;
        }
    };

    wait_for_lock(&db_lock);
    touch(&db_lock);
    os::remove_file(&db);

    match io::mk_file_writer(&db, ~[io::Create]) {
        result::Ok(writer) => {
            writer.write_line(json::to_pretty_str(&json::List(_add_pkg(packages, pkg))));
        }
        result::Err(err) => {
            error(fmt!("failed to dump package db: %s", err));
            os::remove_file(&db_lock);

            return false;
        }
    }

    os::remove_file(&db_lock);

    true
}

pub fn remove_pkg(pkg: &Package) -> bool {
    let root = root();
    let db = root.push(~"db.json");
    let db_lock = root.push(~"db.json.lck");
    let packages = match load_pkgs() {
        result::Ok(packages) => packages,
        result::Err(err) => {
            error(err);

            return false;
        }
    };

    wait_for_lock(&db_lock);
    touch(&db_lock);
    os::remove_file(&db);

    match io::mk_file_writer(&db, ~[io::Create]) {
        result::Ok(writer) => {
            writer.write_line(json::to_pretty_str(&json::List(_rm_pkg(packages, pkg))));
        }
        result::Err(err) => {
            error(fmt!("failed to dump package db: %s", err));
            os::remove_file(&db_lock);

            return false;
        }
    }

    os::remove_file(&db_lock);

    true
}

pub fn compile_input(sysroot: Option<Path>, input: driver::input, dir: &Path,
               flags: ~[~str], cfgs: ~[~str], opt: bool, test: bool) -> bool {
    let lib_dir = dir.push(~"lib");
    let bin_dir = dir.push(~"bin");
    let test_dir = dir.push(~"test");
    let binary = os::args()[0];
    let options: @session::options = @{
        binary: binary,
        crate_type: session::unknown_crate,
        optimize: if opt { session::Aggressive } else { session::No },
        test: test,
        maybe_sysroot: sysroot,
        .. *session::basic_options()
    };
    let sess = driver::build_session(options, diagnostic::emit);
    let cfg = driver::build_configuration(sess, binary, input);
    let mut outputs = driver::build_output_filenames(input, &None, &None,
                                                     sess);
    let {crate, _} = driver::compile_upto(sess, cfg, input, driver::cu_parse,
                                          Some(outputs));

    let mut name = None;
    let mut vers = None;
    let mut uuid = None;
    let mut crate_type = None;

    fn load_link_attr(mis: ~[@ast::meta_item]) -> (Option<~str>,
                                                   Option<~str>,
                                                   Option<~str>) {
        let mut name = None;
        let mut vers = None;
        let mut uuid = None;

        for mis.each |a| {
            match a.node {
                ast::meta_name_value(v, ast::spanned {node: ast::lit_str(s),
                                         span: _}) => {
                    match v {
                        ~"name" => name = Some(*s),
                        ~"vers" => vers = Some(*s),
                        ~"uuid" => uuid = Some(*s),
                        _ => { }
                    }
                }
                _ => {}
            }
        }

        (name, vers, uuid)
    }

    for crate.node.attrs.each |a| {
        match a.node.value.node {
            ast::meta_name_value(v, ast::spanned {node: ast::lit_str(s),
                                     span: _}) => {
                match v {
                    ~"crate_type" => crate_type = Some(*s),
                    _ => {}
                }
            }
            ast::meta_list(v, mis) => {
                match v {
                    ~"link" => {
                        let (n, v, u) = load_link_attr(mis);

                        name = n;
                        vers = v;
                        uuid = u;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    if name.is_none() || vers.is_none() || uuid.is_none() {
        error(~"link attr without (name, vers, uuid) values");

        return false;
    }

    let name = name.get();
    let vers = vers.get();
    let uuid = uuid.get();

    let is_bin = match crate_type {
        Some(crate_type) => {
            match crate_type {
                ~"bin" => true,
                ~"lib" => false,
                _ => {
                    warn(~"unknown crate_type, falling back to lib");

                    false
                }
            }
        }
        None => {
            warn(~"missing crate_type attr, assuming lib");

            false
        }
    };

    if test {
        need_dir(&test_dir);

        outputs = driver::build_output_filenames(input, &Some(test_dir),
                                                 &None, sess)
    }
    else if is_bin {
        need_dir(&bin_dir);

        let path = bin_dir.push(fmt!("%s-%s-%s%s", name,
                                                   hash(name + uuid + vers),
                                                   vers, exe_suffix()));
        outputs = driver::build_output_filenames(input, &None, &Some(path), sess);
    } else {
        need_dir(&lib_dir);

        outputs = driver::build_output_filenames(input, &Some(lib_dir),
                                                 &None, sess)
    }

    driver::compile_rest(sess, cfg, driver::cu_everything,
                         Some(outputs), Some(crate));

    true
}

#[cfg(windows)]
pub fn exe_suffix() -> ~str { ~".exe" }

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn exe_suffix() -> ~str { ~"" }

pub fn compile_crate(sysroot: Option<Path>, crate: &Path, dir: &Path, flags: ~[~str],
               cfgs: ~[~str], opt: bool, test: bool) -> bool {
    compile_input(sysroot, driver::file_input(*crate), dir, flags, cfgs, opt, test)
}

pub fn compile_str(sysroot: Option<Path>, code: ~str, dir: &Path, flags: ~[~str],
                   cfgs: ~[~str], opt: bool, test: bool) -> bool {
    compile_input(sysroot, driver::str_input(code), dir, flags, cfgs, opt, test)
}

#[cfg(windows)]
pub fn link_exe(_src: &Path, _dest: &Path) -> bool{
    /* FIXME: Investigate how to do this on win32
       Node wraps symlinks by having a .bat,
       but that won't work with minGW. */

    false
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn link_exe(src: &Path, dest: &Path) -> bool unsafe {
    do str::as_c_str(src.to_str()) |src_buf| {
        do str::as_c_str(dest.to_str()) |dest_buf| {
            libc::link(src_buf, dest_buf) == 0 as libc::c_int &&
            libc::chmod(dest_buf, 755) == 0 as libc::c_int
        }
    }
}

#[test]
fn test_is_cmd() {
    assert is_cmd(~"build");
    assert is_cmd(~"clean");
    assert is_cmd(~"do");
    assert is_cmd(~"install");
    assert is_cmd(~"prefer");
    assert is_cmd(~"test");
    assert is_cmd(~"uninstall");
    assert is_cmd(~"unprefer");
}

#[test]
fn test_parse_name() {
    assert parse_name(~"org.mozilla.servo").get() == ~"servo";
    assert parse_name(~"org. mozilla.servo 2131").is_err();
}
