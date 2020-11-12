// compile-flags: -O -Zunsound-mir-opts

// EMIT_MIR unneeded_deref_do_not_opt.dont_opt.UnneededDeref.diff
// do not optimize a sequence looking like this:
// _1 = &_2;
// _1 = _3;
// _4 = *_1;
// as the _1 = _3 assignment makes it not legal to replace the last statement with _4 = _2
fn dont_opt() -> u64 {
    let y = 5;
    let _ref = &y;
    let x = 5;
    let mut _1 = &x;
    _1 = _ref;
    let _4 = *_1;
    0
}

// EMIT_MIR unneeded_deref_do_not_opt.do_not_miscompile.UnneededDeref.diff
fn do_not_miscompile() {
    let x = 42;
    let a = 99;
    let mut y = &x;
    let z = &mut y;
    *z = &a;
    assert!(*y == 99);
}

// EMIT_MIR unneeded_deref_do_not_opt.do_not_miscompile_mut_ref.UnneededDeref.diff
// See #78192
fn do_not_miscompile_mut_ref() {
    let a = 1u32;
    let b = 2u32;

    let mut c: *const u32 = &a;
    let d: &u32 = &b;

    let x = unsafe { &*c };
    c = d;
    let z = *x;
}

// EMIT_MIR unneeded_deref_do_not_opt.do_not_use_moved.UnneededDeref.diff
fn do_not_use_moved<T>(x: T) {
    let b = x;
    let z = &b;
}

// EMIT_MIR unneeded_deref_do_not_opt.do_not_opt_different_bbs.UnneededDeref.diff
// We cannot know whether z should be 5 or 33
fn do_not_opt_different_bbs(input: bool) -> u64 {
    let x = 5;
    let y = if input { &x } else { &33 };
    let z = *y;
    z
}

fn main() {
    dont_opt();
    do_not_miscompile();
    do_not_miscompile_mut_ref();
    do_not_use_moved(String::new());
    do_not_opt_different_bbs(false);
}
