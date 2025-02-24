// This is a non-regression test about differences in scopes computed by NLLs and `-Zpolonius=next`
// found during the crater run for PR #117593.
//
// Live loans were computed too early compared to some of the liveness data coming from later passes
// than `liveness::trace`, on some specific CFGs shapes: a variable was dead during tracing but its
// regions were marked live later, and live loans were not recomputed at this point.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius_next polonius
//@ check-pass
//@ [polonius_next] compile-flags: -Z polonius=next
//@ [polonius] compile-flags: -Z polonius

// minimized from wavefc-cli-3.0.0
fn repro1() {
    let a = 0;
    let closure = || {
        let _b = a;
    };

    let callback = if true { Some(closure) } else { None };
    do_it(callback);
}
fn do_it<F>(_: Option<F>)
where
    F: Fn(),
{
}

// minimized from simple-server-0.4.0
fn repro2() {
    let mut a = &();
    let s = S(&mut a);
    let _ = if true { Some(s) } else { None };
}
struct S<'a>(&'a mut &'a ());

// minimized from https://github.com/SHaaD94/AICup2022
fn repro3() {
    let runner = ();
    let writer = debug_interface(&runner);
    let _ = if true { Some(writer) } else { None };
}
fn debug_interface(_: &()) -> &mut dyn std::io::Write {
    unimplemented!()
}

fn main() {}
