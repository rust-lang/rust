// rustfmt-indent_style: Block

#![feature(pub_restricted)]

mod inner_mode {
    pub(super) fn func_name(abc: i32) -> i32 {
        abc
    }
}
