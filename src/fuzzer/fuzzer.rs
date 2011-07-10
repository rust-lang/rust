use std;
use rustc;

import std::fs;
import std::getopts;
import std::getopts::optopt;
import std::getopts::opt_present;
import std::getopts::opt_str;
import std::io;
import std::io::stdout;
import std::vec;
import std::ivec;
import std::str;
import std::uint;

import rustc::syntax::ast;
import rustc::syntax::fold;
import rustc::syntax::walk;
import rustc::syntax::codemap;
import rustc::syntax::print::pprust;

import driver = rustc::driver::rustc; // see https://github.com/graydon/rust/issues/624
import rustc::back::link;
import rustc::driver::rustc::time;
import rustc::driver::session;

/*
// Imports for "the rest of driver::compile_input"
import rustc::metadata::creader;
import rustc::metadata::cstore;
import rustc::syntax::parse::parser;
import rustc::syntax::parse::token;
import rustc::front;
import rustc::front::attr;
import rustc::middle;
import rustc::middle::trans;
import rustc::middle::resolve;
import rustc::middle::ty;
import rustc::middle::typeck;
import rustc::middle::tstate::ck;
import rustc::syntax::print::pp;
import rustc::util::ppaux;
import rustc::lib::llvm;
*/

fn file_contains(&str filename, &str needle) -> bool {
    auto r = io::file_reader(filename);
    auto contents = str::unsafe_from_bytes(r.read_whole_stream());
    ret str::find(contents, needle) != -1;
}

fn find_rust_files(&mutable str[] files, str path) {
    if (str::ends_with(path, ".rs")) {
        if (file_contains(path, "xfail-stage1")) {
            //log_err "Skipping " + path + " because it is marked as xfail-stage1";
        } else if (
            !str::ends_with(path, "constrained-type.rs") &&     // https://github.com/graydon/rust/issues/653
             str::find(path, "utf8") != -1 &&  // https://github.com/graydon/rust/issues/654
             true) {
            //log_err "Skipping " + path + " because of a known bug";
        } else {
            files += ~[path];
        }
    } else if (fs::file_is_dir(path) && str::find(path, "compile-fail") == -1) {
        for (str p in fs::list_dir(path)) {
            find_rust_files(files, p);
        }
    }
}

fn steal_exprs(&ast::crate crate) -> ast::expr[] {
    let @mutable ast::expr[] exprs = @mutable ~[];
    // "Stash" cannot be type-parameterized because of https://github.com/graydon/rust/issues/375
    fn stash_expr(@mutable ast::expr[] es, &@ast::expr e) { *es += ~[*e]; }
    auto v = rec(visit_expr_pre = bind stash_expr(exprs, _) with walk::default_visitor());
    walk::walk_crate(v, crate);
    *exprs
}

// https://github.com/graydon/rust/issues/652
fn safe_to_replace(ast::expr_ e) -> bool {
    alt (e) {
        case (ast::expr_if(_, _, _)) { false }
        case (ast::expr_block(_)) { false }
        case (_) { true }
    }
}

// Replace the |i|th expr (in fold order) of |crate| with |newexpr|.
fn replace_expr_in_crate(&ast::crate crate, uint i, ast::expr_ newexpr) -> ast::crate {
    let @mutable uint j = @mutable 0u;
    fn fold_expr_rep(@mutable uint j_, uint i_, &ast::expr_ newexpr_, &ast::expr_ original, fold::ast_fold fld) -> ast::expr_ {
      *j_ += 1u;
      if (i_ + 1u == *j_ && safe_to_replace(original)) {
        newexpr_
      } else {
        fold::noop_fold_expr(original, fld)
      }
    }
    auto afp = rec(fold_expr = bind fold_expr_rep(j, i, newexpr, _, _) with *fold::default_ast_fold());
    auto af = fold::make_fold(afp);
    let @ast::crate crate2 = @af.fold_crate(crate);
    fold::dummy_out(af); // work around a leak (https://github.com/graydon/rust/issues/651)
    *crate2
}

iter under(uint n) -> uint { let uint i = 0u; while (i < n) { put i; i += 1u; } }

fn devnull() -> io::writer { std::io::string_writer().get_writer() }

fn pp_variants(&ast::crate crate, &session::session sess, &str filename) {
    auto exprs = steal_exprs(crate);
    auto exprsL = ivec::len(exprs);
    if (exprsL < 100u) {
        for each (uint i in under(uint::min(exprsL, 20u))) {
            log_err "Replacing... " + pprust::expr_to_str(@exprs.(i));
            for each (uint j in under(uint::min(exprsL, 5u))) {
                log_err "With... " + pprust::expr_to_str(@exprs.(j));
                auto crate2 = @replace_expr_in_crate(crate, i, exprs.(j).node);
                pprust::print_crate(sess.get_codemap(), crate2, filename, devnull(), pprust::no_ann());
            }
        }
    }
}

fn main(vec[str] args) {
    auto files = ~[];
    auto root = "/Users/jruderman/code/rust/src/"; // XXX
    find_rust_files(files, root); // not using time here because that currently screws with passing-a-mutable-array
    log_err uint::str(ivec::len(files)) + " files";

    auto binary = vec::shift[str](args);
    auto binary_dir = fs::dirname(binary);

    let @session::options sopts =
        @rec(library=false,
             static=false,
             optimize=0u,
             debuginfo=false,
             verify=true,
             run_typestate=true,
             save_temps=false,
             stats=false,
             time_passes=false,
             time_llvm_passes=false,
             output_type=link::output_type_bitcode,
             library_search_paths=[binary_dir + "/lib"],
             sysroot=driver::get_default_sysroot(binary),
             cfg=~[],
             test=false);

    for (str file in files) {
        log_err "=== " + file + " ===";
        let session::session sess = driver::build_session(sopts);
        let @ast::crate crate = time(true, "parsing " + file, bind driver::parse_input(sess, ~[], file));
        pprust::print_crate(sess.get_codemap(), crate, file, devnull(), pprust::no_ann());
        pp_variants(*crate, sess, file);
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
