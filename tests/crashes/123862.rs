//@ known-bug: #123862
//@ needs-rustc-debug-assertions

macro_rules! pos {
    () => {
        (file![$($pos,)* pos!()], line!())
    };
}

fn outer() {
    inner_inlined(main_pos, pos!());
}

fn inner_inlined() {}
