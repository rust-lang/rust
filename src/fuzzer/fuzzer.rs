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
import std::str;
import std::uint;
import std::option;

import rustc::syntax::ast;
import rustc::syntax::fold;
import rustc::syntax::visit;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;
import rustc::syntax::print::pprust;

fn write_file(filename: str, content: str) {
    io::file_writer(filename, [io::create, io::truncate]).write_str(content);
    // Work around https://github.com/graydon/rust/issues/726
    std::run::run_program("chmod", ["644", filename]);
}

fn file_contains(filename: str, needle: str) -> bool {
    let contents = io::read_whole_file_str(filename);
    ret str::find(contents, needle) != -1;
}

fn contains(haystack: str, needle: str) -> bool {
    str::find(haystack, needle) != -1
}

fn find_rust_files(&files: [str], path: str) {
    if str::ends_with(path, ".rs") {
        if file_contains(path, "xfail-test") {
            //log_err "Skipping " + path + " because it is marked as xfail-test";
        } else { files += [path]; }
    } else if fs::file_is_dir(path)
        && str::find(path, "compile-fail") == -1 {
        for p in fs::list_dir(path) {
            find_rust_files(files, p);
        }
    }
}

fn safe_to_steal_expr(e: @ast::expr) -> bool {
    alt e.node {

      // https://github.com/graydon/rust/issues/890
      ast::expr_lit(lit) {
        alt lit.node {
          ast::lit_str(_) { true }
          ast::lit_char(_) { true }
          ast::lit_int(_) { false }
          ast::lit_uint(_) { true }
          ast::lit_mach_int(_, _) { false }
          ast::lit_float(_) { false }
          ast::lit_mach_float(_, _) { false }
          ast::lit_nil. { true }
          ast::lit_bool(_) { true }
        }
      }

      // https://github.com/graydon/rust/issues/890
      ast::expr_cast(_, _) { false }
      ast::expr_assert(_) { false }
      ast::expr_binary(_, _, _) { false }
      ast::expr_assign(_, _) { false }
      ast::expr_assign_op(_, _, _) { false }

      // https://github.com/graydon/rust/issues/764
      ast::expr_fail(option::none.) { false }
      ast::expr_ret(option::none.) { false }
      ast::expr_put(option::none.) { false }

      // These prefix-operator keywords are not being parenthesized when in callee positions.
      // https://github.com/graydon/rust/issues/891
      ast::expr_ret(_) { false }
      ast::expr_put(_) { false }
      ast::expr_check(_, _) { false }
      ast::expr_log(_, _) { false }

      _ { true }
    }
}

fn safe_to_steal_ty(t: @ast::ty) -> bool {
    // Same restrictions
    safe_to_replace_ty(t.node)
}

// Not type-parameterized: https://github.com/graydon/rust/issues/898
fn stash_expr_if(c: fn(@ast::expr)->bool, es: @mutable [ast::expr], e: @ast::expr) {
    if c(e) {
        *es += [*e];
    } else {/* now my indices are wrong :( */ }
}

fn stash_ty_if(c: fn(@ast::ty)->bool, es: @mutable [ast::ty], e: @ast::ty) {
    if c(e) {
        *es += [*e];
    } else {/* now my indices are wrong :( */ }
}

type stolen_stuff = {exprs: [ast::expr], tys: [ast::ty]};

fn steal(crate: ast::crate) -> stolen_stuff {
    let exprs = @mutable [];
    let tys = @mutable [];
    let v = visit::mk_simple_visitor(@{
        visit_expr: bind stash_expr_if(safe_to_steal_expr, exprs, _),
        visit_ty: bind stash_ty_if(safe_to_steal_ty, tys, _)
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(crate, (), v);
    {exprs: *exprs, tys: *tys}
}

// https://github.com/graydon/rust/issues/652
fn safe_to_replace_expr(e: ast::expr_) -> bool {
    alt e {
      ast::expr_if(_, _, _) { false }
      ast::expr_block(_) { false }
      _ { true }
    }
}

fn safe_to_replace_ty(t: ast::ty_) -> bool {
    alt t {
      ast::ty_infer. { false } // always implicit, always top level
      ast::ty_bot. { false }   // in source, can only appear as the out type of a function
      ast::ty_mac(_) { false }
      _ { true }
    }
}

// Replace the |i|th expr (in fold order) of |crate| with |newexpr|.
fn replace_expr_in_crate(crate: ast::crate, i: uint, newexpr: ast::expr) ->
   ast::crate {
    let j: @mutable uint = @mutable 0u;
    fn fold_expr_rep(j_: @mutable uint, i_: uint, newexpr_: ast::expr_,
                     original: ast::expr_, fld: fold::ast_fold) ->
       ast::expr_ {
        *j_ += 1u;
        if i_ + 1u == *j_ && safe_to_replace_expr(original) {
            newexpr_
        } else { fold::noop_fold_expr(original, fld) }
    }
    let afp =
        {fold_expr: bind fold_expr_rep(j, i, newexpr.node, _, _)
            with *fold::default_ast_fold()};
    let af = fold::make_fold(afp);
    let crate2: @ast::crate = @af.fold_crate(crate);
    fold::dummy_out(af); // work around a leak (https://github.com/graydon/rust/issues/651)
    *crate2
}

// Replace the |i|th ty (in fold order) of |crate| with |newty|.
fn replace_ty_in_crate(crate: ast::crate, i: uint, newty: ast::ty) ->
   ast::crate {
    let j: @mutable uint = @mutable 0u;
    fn fold_ty_rep(j_: @mutable uint, i_: uint, newty_: ast::ty_,
                     original: ast::ty_, fld: fold::ast_fold) ->
       ast::ty_ {
        *j_ += 1u;
        if i_ + 1u == *j_ && safe_to_replace_ty(original) {
            newty_
        } else { fold::noop_fold_ty(original, fld) }
    }
    let afp =
        {fold_ty: bind fold_ty_rep(j, i, newty.node, _, _)
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

fn devnull() -> io::writer { std::io::string_writer().get_writer() }

fn as_str(f: fn(io::writer)) -> str {
    let w = std::io::string_writer();
    f(w.get_writer());
    ret w.get_str();
}

fn check_variants_of_ast(crate: ast::crate, codemap: codemap::codemap,
                         filename: str) {
    let stolen = steal(crate);
    check_variants_T(crate, codemap, filename, "expr", stolen.exprs, pprust::expr_to_str, replace_expr_in_crate);
    check_variants_T(crate, codemap, filename, "ty", stolen.tys, pprust::ty_to_str, replace_ty_in_crate);
}

fn check_variants_T<T>(
  crate: ast::crate,
  codemap: codemap::codemap,
  filename: str,
  thing_label: str,
  things: [T],
  stringifier: fn(@T) -> str,
  replacer: fn(ast::crate, uint, T) -> ast::crate
  ) {
    log_err #fmt("%s contains %u %s objects", filename, vec::len(things), thing_label);

    let L = vec::len(things);

    if L < 100u {
        for each i: uint in under(uint::min(L, 20u)) {
            log_err "Replacing... " + stringifier(@things[i]);
            for each j: uint in under(uint::min(L, 5u)) {
                log_err "With... " + stringifier(@things[j]);
                let crate2 = @replacer(crate, i, things[j]);
                // It would be best to test the *crate* for stability, but testing the
                // string for stability is easier and ok for now.
                let str3 =
                    as_str(bind pprust::print_crate(codemap, crate2,
                                                    filename,
                                                    io::string_reader(""), _,
                                                    pprust::no_ann()));
                check_roundtrip_convergence(str3, 1u);
                //let file_label = #fmt("buggy_%s_%s_%u_%u.rs", last_part(filename), thing_label, i, j);
                //check_whole_compiler(str3, file_label);
            }
        }
    }
}

fn last_part(filename: str) -> str {
  let ix = str::rindex(filename, 47u8 /* '/' */);
  assert ix >= 0;
  str::slice(filename, ix as uint + 1u, str::byte_len(filename) - 3u)
}

tag compile_result { known_bug(str); passed(str); failed(str); }

// We'd find more bugs if we could take an AST here, but
// - that would find many "false positives" or unimportant bugs
// - that would be tricky, requiring use of tasks or serialization or randomness.
// This seems to find plenty of bugs as it is :)
fn check_whole_compiler(code: str, suggested_filename: str) {
    let filename = "test.rs";
    write_file(filename, code);
    alt check_whole_compiler_inner(filename) {
      known_bug(s) {
        log_err "Ignoring known bug: " + s;
      }
      failed(s) {
        log_err "check_whole_compiler failure: " + s;
        write_file(suggested_filename, code);
        log_err "Saved as: " + suggested_filename;
      }
      passed(_) { }
    }
}

fn check_whole_compiler_inner(filename: str) -> compile_result {
    let p = std::run::program_output(
            "/Users/jruderman/code/rust/build/stage1/rustc",
            ["-c", filename]);

    //log_err #fmt("Status: %d", p.status);
    if p.err != "" {
        if contains(p.err, "May only branch on boolean predicates") {
            known_bug("https://github.com/graydon/rust/issues/892")
        } else if contains(p.err, "(S->getType()->isPointerTy() && \"Invalid cast\")") {
            known_bug("https://github.com/graydon/rust/issues/895")
        } else if contains(p.err, "Initializer type must match GlobalVariable type") {
            known_bug("https://github.com/graydon/rust/issues/899")
        } else if contains(p.err, "(castIsValid(op, S, Ty) && \"Invalid cast!\"), function Create") {
            known_bug("https://github.com/graydon/rust/issues/901")
        } else {
            log_err "Stderr: " + p.err;
            failed("Unfamiliar error message")
        }
    } else if p.status == 256 {
        if contains(p.out, "Out of stack space, sorry") {
            known_bug("Recursive types - https://github.com/graydon/rust/issues/742")
        } else {
            log_err "Stdout: " + p.out;
            failed("Unfamiliar sudden exit")
        }
    } else if p.status == 6 {
        if contains(p.out, "get_id_ident: can't find item in ext_map") {
            known_bug("https://github.com/graydon/rust/issues/876")
        } else if contains(p.out, "Assertion !cx.terminated failed") {
            known_bug("https://github.com/graydon/rust/issues/893 or https://github.com/graydon/rust/issues/894")
        } else if !contains(p.out, "error:") {
            log_err "Stdout: " + p.out;
            failed("Rejected the input program without a span-error explanation")
        } else {
            passed("Rejected the input program cleanly")
        }
    } else if p.status == 11 {
        failed("Crashed!?")
    } else if p.status == 0 {
        passed("Accepted the input program")
    } else {
        log_err p.status;
        log_err "!Stdout: " + p.out;
        failed("Unfamiliar status code")
    }
}


fn parse_and_print(code: str) -> str {
    let filename = "tmp.rs";
    let sess = @{cm: codemap::new_codemap(), mutable next_id: 0};
    //write_file(filename, code);
    let crate = parser::parse_crate_from_source_str(
        filename, code, [], sess);
    ret as_str(bind pprust::print_crate(sess.cm, crate,
                                        filename,
                                        io::string_reader(code), _,
                                        pprust::no_ann()));
}

fn content_is_dangerous_to_modify(code: str) -> bool {
    let dangerous_patterns =
        ["#macro", // not safe to steal things inside of it, because they have a special syntax
         "#",      // strange representation of the arguments to #fmt, for example
         "tag",    // typeck hang: https://github.com/graydon/rust/issues/900
         " be "];  // don't want to replace its child with a non-call: "Non-call expression in tail call"

    for p: str in dangerous_patterns { if contains(code, p) { ret true; } }
    ret false;
}

fn content_is_confusing(code: str) -> bool {
    let confusing_patterns =
        ["self",       // crazy rules enforced by parser rather than typechecker?
        "spawn",       // precedence issues?
         "bind",       // precedence issues?
         "\n\n\n\n\n"  // https://github.com/graydon/rust/issues/850
        ];

    for p: str in confusing_patterns { if contains(code, p) { ret true; } }
    ret false;
}

fn file_is_confusing(filename: str) -> bool {
    let confusing_files = [];

    for f in confusing_files { if contains(filename, f) { ret true; } }

    ret false;
}

fn check_roundtrip_convergence(code: str, maxIters: uint) {

    let i = 0u;
    let new = code;
    let old = code;

    while i < maxIters {
        old = new;
        if content_is_confusing(old) { ret; }
        new = parse_and_print(old);
        if old == new { break; }
        i += 1u;
    }

    if old == new {
        log_err #fmt["Converged after %u iterations", i];
    } else {
        log_err #fmt["Did not converge after %u iterations!", i];
        write_file("round-trip-a.rs", old);
        write_file("round-trip-b.rs", new);
        std::run::run_program("diff",
                              ["-w", "-u", "round-trip-a.rs",
                               "round-trip-b.rs"]);
        fail "Mismatch";
    }
}

fn check_convergence(files: [str]) {
    log_err #fmt["pp convergence tests: %u files", vec::len(files)];
    for file in files {
        if !file_is_confusing(file) {
            let s = io::read_whole_file_str(file);
            if !content_is_confusing(s) {
                log_err #fmt["pp converge: %s", file];
                // Change from 7u to 2u once https://github.com/graydon/rust/issues/850 is fixed
                check_roundtrip_convergence(s, 7u);
            }
        }
    }
}

fn check_variants(files: [str]) {
    for file in files {
        if !file_is_confusing(file) {
            let s = io::read_whole_file_str(file);
            if content_is_dangerous_to_modify(s) || content_is_confusing(s) {
                cont;
            }
            log_err "check_variants: " + file;
            let sess = @{cm: codemap::new_codemap(), mutable next_id: 0};
            let crate =
                parser::parse_crate_from_source_str(
                    file,
                    s, [], sess);
            log_err as_str(bind pprust::print_crate(sess.cm, crate,
                                                    file,
                                                    io::string_reader(s), _,
                                                    pprust::no_ann()));
            check_variants_of_ast(*crate, sess.cm, file);
        }
    }
}

fn main(args: [str]) {
    if vec::len(args) != 2u {
        log_err #fmt["usage: %s <testdir>", args[0]];
        ret;
    }
    let files = [];
    let root = args[1];

    find_rust_files(files, root);
    check_convergence(files);
    check_variants(files);
    log_err "Fuzzer done";
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
