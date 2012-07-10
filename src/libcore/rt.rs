//! Runtime calls emitted by the compiler.

import libc::c_char;
import libc::size_t;

type rust_task = libc::c_void;

extern mod rustrt {
    #[rust_stack]
    fn rust_upcall_fail(expr: *c_char, file: *c_char, line: size_t);
}

// FIXME (#2861): This needs both the attribute, and the name prefixed with
// 'rt_', otherwise the compiler won't find it. To fix this, see
// gather_rust_rtcalls.
#[rt(fail)]
fn rt_fail(expr: *c_char, file: *c_char, line: size_t) {
    rustrt::rust_upcall_fail(expr, file, line);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
