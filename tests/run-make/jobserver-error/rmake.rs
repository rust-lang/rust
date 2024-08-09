// If the environment variables contain an invalid `jobserver-auth`, this used
// to cause an ICE (internal compiler error) until this was fixed in #109694.
// Proper handling has been added, and this test checks that helpful warnings
// and errors are printed instead in case of a wrong jobserver.
// See https://github.com/rust-lang/rust/issues/46981

// FIXME(Oneirical): The original test included this memo:
// # Note that by default, the compiler uses file descriptors 0 (stdin), 1 (stdout), 2 (stderr),
// # but also 3 and 4 for either end of the ctrl-c signal handler self-pipe.

// FIXME(Oneirical): only-linux ignore-cross-compile

use run_make_support::{diff, rfs, rustc};

fn main() {
    let out = rustc()
        .stdin("fn main() {}")
        .env("MAKEFLAGS", "--jobserver-auth=5,5")
        .run_fail()
        .stderr_utf8();
    diff().expected_file("cannot_open_fd.stderr").actual_text("actual", out).run();
    // FIXME(Oneirical): Find how to use file descriptor "3" with run-make-support
    // and pipe /dev/null into it.
    // Original lines:
    //
    // bash -c 'echo "fn main() {}" | makeflags="--jobserver-auth=3,3" $(rustc) - 3</dev/null' \
    // 2>&1 | diff not_a_pipe.stderr -
    // # this test randomly fails, see https://github.com/rust-lang/rust/issues/110321
    // disabled:
    // bash -c 'echo "fn main() {}" | makeflags="--jobserver-auth=3,3" $(rustc) - \
    // 3< <(cat /dev/null)' 2>&1 | diff poisoned_pipe.stderr -
    //
    // let out = rustc()
    //     .stdin("fn main() {}")
    //     .input("-")
    //     .env("MAKEFLAGS", "--jobserver-auth=0,0")
    //     .run()
    //     .stderr_utf8();
    // diff().expected_file("not_a_pipe.stderr").actual_text("actual", out).run();
}
