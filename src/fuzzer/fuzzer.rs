import std::fs;
import std::getopts;
import std::getopts::optopt;
import std::getopts::opt_present;
import std::getopts::opt_str;
import std::io;
import std::vec;

type src_gen = iter() -> str;

iter dir_src_gen(str dir) -> str {
}

fn usage(str binary) {
    io::stdout().write_line("usage");
}

type session = rec(str srcdir);

fn make_session(vec[str] args) -> session {
    // Directory of rust source files to use as input
    auto opt_src = "src";

    auto binary = vec::shift[str](args);
    auto opts  = [optopt(opt_src)];
    auto match;
    alt (getopts::getopts(args, opts)) {
        case (getopts::failure(?f)) {
            log_err #fmt("error: %s", getopts::fail_str(f));
            fail;
        }
        case (getopts::success(?m)) {
            match = m;
        }
    };

    if (!opt_present(match, opt_src)) {
        usage(binary);
        fail;
    }

    auto srcdir = opt_str(match, opt_src);

    ret rec(srcdir = srcdir);
}

fn log_session(session sess) {
    log #fmt("srcdir: %s", sess.srcdir);
}

fn run_session(session sess) {
}

fn main(vec[str] args) {
    auto sess = make_session(args);
    log_session(sess);
    run_session(sess);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
