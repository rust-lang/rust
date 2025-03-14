// ignore-tidy-linelength
//! If the environment variables contain an invalid `jobserver-auth`, this used to cause an ICE
//! until this was fixed in [do not panic on failure to acquire jobserver token
//! #109694](https://github.com/rust-lang/rust/pull/109694).
//!
//! Proper handling has been added, and this test checks that helpful warnings and errors are
//! printed instead in case of a wrong jobserver. See
//! <https://github.com/rust-lang/rust/issues/46981>.

//@ only-linux
//@ ignore-cross-compile

#![deny(warnings)]

use run_make_support::{diff, rustc};

fn main() {
    let out = rustc()
        .stdin_buf(("fn main() {}").as_bytes())
        .env("MAKEFLAGS", "--jobserver-auth=1000,1000")
        .run_fail()
        .stderr_utf8();
    diff().expected_file("cannot_open_fd.stderr").actual_text("actual", out).run();

    let out = rustc()
        .stdin_buf(("fn main() {}").as_bytes())
        .input("-")
        .env("MAKEFLAGS", "--jobserver-auth=3,3")
        .set_aux_fd(3, std::fs::File::open("/dev/null").unwrap())
        .run()
        .stderr_utf8();
    diff().expected_file("not_a_pipe.stderr").actual_text("actual", out).run();

    // FIXME(#110321): the Makefile version had a disabled check:
    //
    // ```makefile
    // bash -c 'echo "fn main() {}" | MAKEFLAGS="--jobserver-auth=3,3" $(RUSTC) - 3< <(cat /dev/null)' 2>&1 | diff poisoned_pipe.stderr -
    // ```
    //
    // > the jobserver helper thread launched here gets starved out and doesn't run, while the
    // > coordinator thread continually processes work using the implicit jobserver token, never
    // > yielding long enough for the jobserver helper to do its work (and process the error).
    //
    // but is not necessarily worth fixing as it might require changing coordinator behavior that
    // might regress performance. See discussion at
    // <https://github.com/rust-lang/rust/issues/110321#issuecomment-1636914956>.
}
