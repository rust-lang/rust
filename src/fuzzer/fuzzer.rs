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
import std::istr;
import std::uint;
import std::option;

import rustc::syntax::ast;
import rustc::syntax::fold;
import rustc::syntax::visit;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;
import rustc::syntax::print::pprust;

fn write_file(filename: &str, content: &str) {
    io::file_writer(filename, [io::create, io::truncate]).write_str(content);
    // Work around https://github.com/graydon/rust/issues/726
    std::run::run_program("chmod", ["644", filename]);
}

fn file_contains(filename: &str, needle: &str) -> bool {
    let contents = io::read_whole_file_str(filename);
    ret str::find(contents, needle) != -1;
}

fn contains(haystack: &str, needle: &str) -> bool {
    str::find(haystack, needle) != -1
}

fn find_rust_files(files: &mutable [str], path: str) {
    if str::ends_with(path, ".rs") {
        if file_contains(path, "xfail-stage1") {
            //log_err "Skipping " + path + " because it is marked as xfail-stage1";
        } else { files += [path]; }
    } else if fs::file_is_dir(istr::from_estr(path))
        && str::find(path, "compile-fail") == -1 {
        for p in fs::list_dir(istr::from_estr(path)) {
            find_rust_files(files, istr::to_estr(p));
        }
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
      ast::expr_assert(_) { false }
      ast::expr_binary(_, _, _) { false }
      ast::expr_assign(_, _) { false }
      ast::expr_assign_op(_, _, _) { false }
      ast::expr_fail(option::none.) {
        false
        /* https://github.com/graydon/rust/issues/764 */

      }
      ast::expr_ret(option::none.) { false }
      ast::expr_put(option::none.) { false }


      ast::expr_ret(_) {
        false
        /* lots of code generation issues, such as https://github.com/graydon/rust/issues/770 */

      }
      ast::expr_fail(_) { false }


      _ {
        true
      }
    }
}

fn steal_exprs(crate: &ast::crate) -> [ast::expr] {
    let exprs: @mutable [ast::expr] = @mutable [];
    // "Stash" is not type-parameterized because of the need for safe_to_steal
    fn stash_expr(es: @mutable [ast::expr], e: &@ast::expr) {
        if safe_to_steal(e.node) {
            *es += [*e];
        } else {/* now my indices are wrong :( */ }
    }
    let v =
        visit::mk_simple_visitor(@{visit_expr: bind stash_expr(exprs, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_crate(crate, (), v);;
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
    ;
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

fn check_variants_of_ast(crate: &ast::crate, codemap: &codemap::codemap,
                         filename: &str) {
    let exprs = steal_exprs(crate);
    let exprsL = vec::len(exprs);
    if exprsL < 100u {
        for each i: uint in under(uint::min(exprsL, 20u)) {
            log_err "Replacing... " + pprust::expr_to_str(@exprs[i]);
            for each j: uint in under(uint::min(exprsL, 5u)) {
                log_err "With... " + pprust::expr_to_str(@exprs[j]);
                let crate2 = @replace_expr_in_crate(crate, i, exprs[j].node);
                // It would be best to test the *crate* for stability, but testing the
                // string for stability is easier and ok for now.
                let str3 =
                    as_str(bind pprust::print_crate(codemap, crate2, filename,
                                                    io::string_reader(""), _,
                                                    pprust::no_ann()));
                // 1u would be sane here, but the pretty-printer currently has lots of whitespace and paren issues,
                // and https://github.com/graydon/rust/issues/766 is hilarious.
                check_roundtrip_convergence(str3, 7u);
                //check_whole_compiler(str3);
            }
        }
    }
}

// We'd find more bugs if we could take an AST here, but
// - that would find many "false positives" or unimportant bugs
// - that would be tricky, requiring use of tasks or serialization or randomness.
// This seems to find plenty of bugs as it is :)
fn check_whole_compiler(code: &str) {
    let filename = "test.rs";
    write_file(filename, code);
    let p =
        std::run::program_output("/Users/jruderman/code/rust/build/stage1/rustc",
                                 ["-c", filename]);

    //log_err #fmt("Status: %d", p.status);
    //log_err "Output: " + p.out;
    if p.err != "" {
        if contains(p.err, "argument of incompatible type") {
            log_err "https://github.com/graydon/rust/issues/769";
        } else if contains(p.err,
                           "Cannot create binary operator with two operands of differing type")
         {
            log_err "https://github.com/graydon/rust/issues/770";
        } else if contains(p.err, "May only branch on boolean predicates!") {
            log_err "https://github.com/graydon/rust/issues/770 or https://github.com/graydon/rust/issues/776";
        } else if contains(p.err, "Invalid constantexpr cast!") &&
                      contains(code, "!") {
            log_err "https://github.com/graydon/rust/issues/777";
        } else if contains(p.err,
                           "Both operands to ICmp instruction are not of the same type!")
                      && contains(code, "!") {
            log_err "https://github.com/graydon/rust/issues/777 #issuecomment-1678487";
        } else if contains(p.err, "Ptr must be a pointer to Val type!") &&
                      contains(code, "!") {
            log_err "https://github.com/graydon/rust/issues/779";
        } else if contains(p.err, "Calling a function with bad signature!") &&
                      (contains(code, "iter") || contains(code, "range")) {
            log_err "https://github.com/graydon/rust/issues/771 - calling an iter fails";
        } else if contains(p.err, "Calling a function with a bad signature!")
                      && contains(code, "empty") {
            log_err "https://github.com/graydon/rust/issues/775 - possibly a modification of run-pass/import-glob-crate.rs";
        } else if contains(p.err, "Invalid type for pointer element!") &&
                      contains(code, "put") {
            log_err "https://github.com/graydon/rust/issues/773 - put put ()";
        } else if contains(p.err, "pointer being freed was not allocated") &&
                      contains(p.out, "Out of stack space, sorry") {
            log_err "https://github.com/graydon/rust/issues/768 + https://github.com/graydon/rust/issues/778"
        } else {
            log_err "Stderr: " + p.err;
            fail "Unfamiliar error message";
        }
    } else if contains(p.out, "non-exhaustive match failure") &&
                  contains(p.out, "alias.rs") {
        log_err "https://github.com/graydon/rust/issues/772";
    } else if contains(p.out, "non-exhaustive match failure") &&
                  contains(p.out, "trans.rs") && contains(code, "put") {
        log_err "https://github.com/graydon/rust/issues/774";
    } else if contains(p.out, "Out of stack space, sorry") {
        log_err "Possibly a variant of https://github.com/graydon/rust/issues/768";
    } else if p.status == 256 {
        if !contains(p.out, "error:") {
            fail "Exited with status 256 without a span-error";
        }
    } else if p.status == 11 {
        log_err "What is this I don't even";
    } else if p.status != 0 { fail "Unfamiliar status code"; }
}

fn parse_and_print(code: &str) -> str {
    let filename = "tmp.rs";
    let sess = @{cm: codemap::new_codemap(), mutable next_id: 0};
    //write_file(filename, code);
    let crate = parser::parse_crate_from_source_str(filename, code, [], sess);
    ret as_str(bind pprust::print_crate(sess.cm, crate, filename,
                                        io::string_reader(code), _,
                                        pprust::no_ann()));
}

fn content_is_dangerous_to_modify(code: &str) -> bool {
    let dangerous_patterns =
        ["obj", // not safe to steal; https://github.com/graydon/rust/issues/761
         "#macro", // not safe to steal things inside of it, because they have a special syntax
         "#", // strange representation of the arguments to #fmt, for example
         " be ", // don't want to replace its child with a non-call: "Non-call expression in tail call"
         "@"]; // hangs when compiling: https://github.com/graydon/rust/issues/768

    for p: str in dangerous_patterns { if contains(code, p) { ret true; } }
    ret false;
}

fn content_is_confusing(code: &str) ->
   bool { // https://github.com/graydon/rust/issues/671
          // https://github.com/graydon/rust/issues/669
          // https://github.com/graydon/rust/issues/669
          // https://github.com/graydon/rust/issues/669
          // crazy rules enforced by parser rather than typechecker?
          // more precedence issues
          // more precedence issues?

    let confusing_patterns =
        ["#macro", "][]", "][mutable]", "][mutable ]", "self", "spawn",
         "bind", "\n\n\n\n\n", // https://github.com/graydon/rust/issues/759
         " : ", // https://github.com/graydon/rust/issues/760
         "if ret", "alt ret", "if fail", "alt fail"];

    for p: str in confusing_patterns { if contains(code, p) { ret true; } }
    ret false;
}

fn file_is_confusing(filename: &str) -> bool {

    // https://github.com/graydon/rust/issues/674

    // something to do with () as a lone pattern

    // an issue where -2147483648 gains an
    // extra negative sign each time through,
    // which i can't reproduce using "rustc
    // --pretty normal"???
    let confusing_files =
        ["block-expr-precedence.rs", "nil-pattern.rs",
         "syntax-extension-fmt.rs",
         "newtype.rs"]; // modifying it hits something like https://github.com/graydon/rust/issues/670

    for f in confusing_files { if contains(filename, f) { ret true; } }

    ret false;
}

fn check_roundtrip_convergence(code: &str, maxIters: uint) {

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

fn check_convergence(files: &[str]) {
    log_err #fmt["pp convergence tests: %u files", vec::len(files)];
    for file in files {
        if !file_is_confusing(file) {
            let s = io::read_whole_file_str(file);
            if !content_is_confusing(s) {
                log_err #fmt["pp converge: %s", file];
                // Change from 7u to 2u when https://github.com/graydon/rust/issues/759 is fixed
                check_roundtrip_convergence(s, 7u);
            }
        }
    }
}

fn check_variants(files: &[str]) {
    for file in files {
        if !file_is_confusing(file) {
            let s = io::read_whole_file_str(file);
            if content_is_dangerous_to_modify(s) || content_is_confusing(s) {
                cont;
            }
            log_err "check_variants: " + file;
            let sess = @{cm: codemap::new_codemap(), mutable next_id: 0};
            let crate =
                parser::parse_crate_from_source_str(file, s, [], sess);
            log_err as_str(bind pprust::print_crate(sess.cm, crate, file,
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
