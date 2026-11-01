//@ known-bug: #131787
#[track_caller]
static no_mangle: u32 = {
    unimplemented!();
};
