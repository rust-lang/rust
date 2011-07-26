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

fn read_whole_file(&str filename) -> str {
    str::unsafe_from_bytes_ivec(ioivec::file_reader(filename).read_whole_stream())
}

fn write_file(&str filename, &str content) {
    ioivec::file_writer(filename, ~[ioivec::create,
                                    ioivec::truncate]).write_str(content);
}

fn file_contains(&str filename, &str needle) -> bool {
    auto contents = read_whole_file(filename);
    ret str::find(contents, needle) != -1;
}

fn contains(&str haystack, &str needle) -> bool { str::find(haystack, needle) != -1 }

fn find_rust_files(&mutable str[] files, str path) {
    if (str::ends_with(path, ".rs")) {
        if (file_contains(path, "xfail-stage1")) {
            //log_err "Skipping " + path + " because it is marked as xfail-stage1";
        } else {
            files += ~[path];
        }
    } else if (fs::file_is_dir(path) && str::find(path, "compile-fail") == -1) {
        for (str p in fs::list_dir(path)) {
            find_rust_files(files, p);
        }
    }
}

fn safe_to_steal(ast::expr_ e) -> bool {
    alt (e) {
        // pretty-printer precedence issues -- https://github.com/graydon/rust/issues/670
        case (ast::expr_unary(_, _)) { false }
        case (ast::expr_lit(?lit)) {
            alt(lit.node) {
                case(ast::lit_str(_, _)) { true }
                case(ast::lit_char(_)) { true }
                case(ast::lit_int(_)) { false }
                case(ast::lit_uint(_)) { false }
                case(ast::lit_mach_int(_, _)) { false }
                case(ast::lit_float(_)) { false }
                case(ast::lit_mach_float(_, _)) { false }
                case(ast::lit_nil) { true }
                case(ast::lit_bool(_)) { true }
            }
        }
        case (ast::expr_cast(_, _)) { false }
        case (ast::expr_send(_, _)) { false }
        case (ast::expr_recv(_, _)) { false }
        case (ast::expr_assert(_)) { false }
        case (ast::expr_binary(_, _, _)) { false }
        case (ast::expr_assign(_, _)) { false }
        case (ast::expr_assign_op(_, _, _)) { false }

        // https://github.com/graydon/rust/issues/676
        case (ast::expr_ret(option::none)) { false }
        case (ast::expr_put(option::none)) { false }

        case (_) { true }
    }
}

fn steal_exprs(&ast::crate crate) -> ast::expr[] {
    let @mutable ast::expr[] exprs = @mutable ~[];
    // "Stash" is not type-parameterized because of the need for safe_to_steal
    fn stash_expr(@mutable ast::expr[] es, &@ast::expr e) { if (safe_to_steal(e.node)) { *es += ~[*e]; } else { /* now my indices are wrong :( */ } }
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

fn devnull() -> ioivec::writer { std::ioivec::string_writer().get_writer() }

fn as_str(fn (ioivec::writer) f) -> str {
    auto w = std::ioivec::string_writer();
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

fn parse_and_print(&str code) -> str {
    auto filename = "";
    auto codemap = codemap::new_codemap();
    auto crate = parser::parse_crate_from_source_str(filename, code, ~[], codemap);
    ret as_str(bind pprust::print_crate(codemap, crate, filename,
                                        ioivec::string_reader(code),
                                        _, pprust::no_ann()));
}

fn content_is_confusing(&str code) -> bool {
    auto confusing_patterns = [
        "#macro",      // https://github.com/graydon/rust/issues/671
        "][]",         // https://github.com/graydon/rust/issues/669
        "][mutable]",  // https://github.com/graydon/rust/issues/669
        "][mutable ]", // https://github.com/graydon/rust/issues/669
        "self",        // crazy rules enforced by parser rather than typechecker?
        "spawn",       // more precedence issues
        "bind"         // more precedence issues?
    ];

    for (str p in confusing_patterns) {
        if contains(code, p) {
            ret true;
        }
    }
    ret false;
}

fn file_is_confusing(&str filename) -> bool {
    auto confusing_files = [

        "block-expr-precedence.rs",  // https://github.com/graydon/rust/issues/674

        "syntax-extension-fmt.rs"    // an issue where -2147483648 gains an
                                     // extra negative sign each time through,
                                     // which i can't reproduce using "rustc
                                     // --pretty normal"???
    ];

    for (str f in confusing_files) {
        if contains(filename, f) {
            ret true;
        }
    }

    ret false;
}

fn check_roundtrip_convergence(&str code) {

    auto i = 0;
    auto new = code;
    auto old = code;

    while (i < 10) {
        old = new;
        new = parse_and_print(old);
        i += 1;
        log_err #fmt("cycle %d", i);
    }

    if old != new {
        write_file("round-trip-a.rs", old);
        write_file("round-trip-b.rs", new);
        std::run::run_program("kdiff3", ["round-trip-a.rs", "round-trip-b.rs"]);
        fail "Mismatch";
    }
}

fn main(vec[str] args) {
    if (vec::len(args) != 2u) {
        log_err #fmt("usage: %s <testdir>", args.(0));
        ret;
    }
    auto files = ~[];
    auto root = args.(1);
    find_rust_files(files, root); // not using time here because that currently screws with passing-a-mutable-array
    log_err #fmt("%u files", ivec::len(files));

    for (str file in files) {
        log_err "=== " + file + " ===";
        if ! file_is_confusing(file) {
            auto s = read_whole_file(file);
            if ! content_is_confusing(s) {
                check_roundtrip_convergence(s);
            }
        }

        //pprust::print_crate(cm, crate, file, devnull(), pprust::no_ann());
        // Currently hits https://github.com/graydon/rust/issues/675
        //pp_variants(*crate, cm, file);
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
