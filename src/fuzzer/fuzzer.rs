use std;
use rustc;

import std::fs;
import std::getopts;
import std::getopts::optopt;
import std::getopts::opt_present;
import std::getopts::opt_str;
import std::ioivec;
import std::ioivec::stdout;
import std::vec;
import std::ivec;
import std::str;
import std::uint;
import std::option;

import rustc::syntax::ast;
import rustc::syntax::fold;
import rustc::syntax::walk;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;
import rustc::syntax::print::pprust;

/*
// Imports for "the rest of driver::compile_input"
import driver = rustc::driver::rustc; // see https://github.com/graydon/rust/issues/624
import rustc::back::link;
import rustc::driver::rustc::time;
import rustc::driver::session;

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

fn read_whole_file(filename: &str) -> str {
    str::unsafe_from_bytes_ivec(ioivec::file_reader(filename).read_whole_stream())
}

fn write_file(filename: &str, content: &str) {
    ioivec::file_writer(filename,
                        ~[ioivec::create,
                          ioivec::truncate]).write_str(content);
}

fn file_contains(filename: &str, needle: &str) -> bool {
    let contents = read_whole_file(filename);
    ret str::find(contents, needle) != -1;
}

fn contains(haystack: &str, needle: &str) -> bool {
    str::find(haystack, needle) != -1
}

fn find_rust_files(files: &mutable str[], path: str) {
    if str::ends_with(path, ".rs") {
        if file_contains(path, "xfail-stage1") {
            //log_err "Skipping " + path + " because it is marked as xfail-stage1";
        } else { files += ~[path]; }
    } else if (fs::file_is_dir(path) && str::find(path, "compile-fail") == -1)
     {
        for p: str  in fs::list_dir(path) { find_rust_files(files, p); }
    }
}

fn safe_to_steal(e: ast::expr_) -> bool {
    alt e {

      // pretty-printer precedence issues -- https://github.com/graydon/rust/issues/670
      ast::expr_unary(_, _) {
        false
      }
      ast::expr_lit(lit) {
        alt lit.node {
          ast::lit_str(_, _) { true }
          ast::lit_char(_) { true }
          ast::lit_int(_) { false }
          ast::lit_uint(_) { false }
          ast::lit_mach_int(_, _) { false }
          ast::lit_float(_) { false }
          ast::lit_mach_float(_, _) { false }
          ast::lit_nil. { true }
          ast::lit_bool(_) { true }
        }
      }
      ast::expr_cast(_, _) { false }
      ast::expr_send(_, _) { false }
      ast::expr_recv(_, _) { false }
      ast::expr_assert(_) { false }
      ast::expr_binary(_, _, _) { false }
      ast::expr_assign(_, _) { false }
      ast::expr_assign_op(_, _, _) { false }


      // https://github.com/graydon/rust/issues/676
      ast::expr_ret(option::none.) {
        false
      }
      ast::expr_put(option::none.) { false }


      _ {
        true
      }
    }
}

fn steal_exprs(crate: &ast::crate) -> ast::expr[] {
    let exprs: @mutable ast::expr[] = @mutable ~[];
    // "Stash" is not type-parameterized because of the need for safe_to_steal
    fn stash_expr(es: @mutable ast::expr[], e: &@ast::expr) {
        if safe_to_steal(e.node) {
            *es += ~[*e];
        } else {/* now my indices are wrong :( */ }
    }
    let v =
        {visit_expr_pre: bind stash_expr(exprs, _)
            with walk::default_visitor()};
    walk::walk_crate(v, crate);
    *exprs
}

// https://github.com/graydon/rust/issues/652
fn safe_to_replace(e: ast::expr_) -> bool {
    alt e {
      ast::expr_if(_, _, _) { false }
      ast::expr_block(_) { false }
      _ { true }
    }
}

// Replace the |i|th expr (in fold order) of |crate| with |newexpr|.
fn replace_expr_in_crate(crate: &ast::crate, i: uint, newexpr: ast::expr_) ->
   ast::crate {
    let j: @mutable uint = @mutable 0u;
    fn fold_expr_rep(j_: @mutable uint, i_: uint, newexpr_: &ast::expr_,
                     original: &ast::expr_, fld: fold::ast_fold) ->
       ast::expr_ {
        *j_ += 1u;
        if i_ + 1u == *j_ && safe_to_replace(original) {
            newexpr_
        } else { fold::noop_fold_expr(original, fld) }
    }
    let afp =
        {fold_expr: bind fold_expr_rep(j, i, newexpr, _, _)
            with *fold::default_ast_fold()};
    let af = fold::make_fold(afp);
    let crate2: @ast::crate = @af.fold_crate(crate);
    fold::dummy_out(af); // work around a leak (https://github.com/graydon/rust/issues/651)
    *crate2
}

iter under(n: uint) -> uint {
    let i: uint = 0u;
    while i < n { put i; i += 1u; }
}

fn devnull() -> ioivec::writer { std::ioivec::string_writer().get_writer() }

fn as_str(f: fn(ioivec::writer) ) -> str {
    let w = std::ioivec::string_writer();
    f(w.get_writer());
    ret w.get_str();
}

/*
fn pp_variants(&ast::crate crate, &codemap::codemap cmap, &str filename) {
    auto exprs = steal_exprs(crate);
    auto exprsL = ivec::len(exprs);
    if (exprsL < 100u) {
        for each (uint i in under(uint::min(exprsL, 20u))) {
            log_err "Replacing... " + pprust::expr_to_str(@exprs.(i));
            for each (uint j in under(uint::min(exprsL, 5u))) {
                log_err "With... " + pprust::expr_to_str(@exprs.(j));
                auto crate2 = @replace_expr_in_crate(crate, i, exprs.(j).node);
                check_roundtrip(crate2, cmap, filename + ".4.rs");
            }
        }
    }
}
*/

fn parse_and_print(code: &str) -> str {
    let filename = "";
    let codemap = codemap::new_codemap();
    let crate =
        parser::parse_crate_from_source_str(filename, code, ~[], codemap);
    ret as_str(bind pprust::print_crate(codemap, crate, filename,
                                        ioivec::string_reader(code), _,
                                        pprust::no_ann()));
}

fn content_is_confusing(code: &str) -> bool {
    let  // https://github.com/graydon/rust/issues/671
         // https://github.com/graydon/rust/issues/669
         // https://github.com/graydon/rust/issues/669
         // https://github.com/graydon/rust/issues/669
         // crazy rules enforced by parser rather than typechecker?
         // more precedence issues
         // more precedence issues?
        confusing_patterns =
        ["#macro", "][]", "][mutable]", "][mutable ]", "self", "spawn",
         "bind"];

    for p: str  in confusing_patterns { if contains(code, p) { ret true; } }
    ret false;
}

fn file_is_confusing(filename: &str) -> bool {
    let 

         // https://github.com/graydon/rust/issues/674

         // something to do with () as a lone pattern

         // an issue where -2147483648 gains an
         // extra negative sign each time through,
         // which i can't reproduce using "rustc
         // --pretty normal"???
         confusing_files =
        ["block-expr-precedence.rs", "nil-pattern.rs",
         "syntax-extension-fmt.rs"];

    for f: str  in confusing_files { if contains(filename, f) { ret true; } }

    ret false;
}

fn check_roundtrip_convergence(code: &str) {

    let i = 0;
    let new = code;
    let old = code;

    while i < 10 {
        old = new;
        new = parse_and_print(old);
        if content_is_confusing(new) { ret; }
        i += 1;
        log #fmt("cycle %d", i);
    }


    if old != new {
        write_file("round-trip-a.rs", old);
        write_file("round-trip-b.rs", new);
        std::run::run_program("kdiff3",
                              ["round-trip-a.rs", "round-trip-b.rs"]);
        fail "Mismatch";
    }
}
fn check_convergence(files: &str[]) {
    log_err #fmt("pp convergence tests: %u files", ivec::len(files));
    for file: str  in files {

        log_err #fmt("pp converge: %s", file);
        if !file_is_confusing(file) {
            let s = read_whole_file(file);
            if !content_is_confusing(s) { check_roundtrip_convergence(s); }
        }

        //pprust::print_crate(cm, crate, file, devnull(), pprust::no_ann());
        // Currently hits https://github.com/graydon/rust/issues/675
        //pp_variants(*crate, cm, file);
    }
}

fn main(args: vec[str]) {
    if vec::len(args) != 2u {
        log_err #fmt("usage: %s <testdir>", args.(0));
        ret;
    }
    let files = ~[];
    let root = args.(1);

    find_rust_files(files, root);
    check_convergence(files);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
