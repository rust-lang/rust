use std;
use rustc;

import std::fs;
import std::getopts;
import std::getopts::optopt;
import std::getopts::opt_present;
import std::getopts::opt_str;
import std::io;
import std::vec;
import std::ivec;
import std::str;

import rustc::back::link;
import rustc::syntax::ast;
import driver = rustc::driver::rustc; // see https://github.com/graydon/rust/issues/624
import rustc::driver::session;


fn find_rust_files(&mutable str[] files, str root) {
    for (str filename in fs::list_dir(root)) {
        if (str::ends_with(filename, ".rs")) {
           files += ~[filename];
        }
    }
}

fn main(vec[str] args) {
    auto files = ~[];
    auto root = "/Users/jruderman/code/rust/src/lib/"; // XXX
    find_rust_files(files, root); // not using driver::time here because that currently screws with passing-a-mutable-array

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

    let session::session sess = driver::build_session(sopts);

    log_err ivec::len(files);
    for (str file in files) {
        log_err file;
        // Can't use parse_input here because of https://github.com/graydon/rust/issues/632 :(
        //auto crate = driver::parse_input(sess, ~[], file);
        //let @ast::crate crate = driver::time(true, "parsing " + file, bind driver::parse_input(sess, ~[], file));
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
