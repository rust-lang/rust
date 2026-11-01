#![feature(link_arg_attribute)]

#[link(kind = "static", name = "native_dep_1", modifiers = "-bundle")]
#[link(kind = "link-arg", name = "some_flag")]
#[link(kind = "static", name = "native_dep_2", modifiers = "-bundle")]
extern "C" {
    pub fn foo();
}

pub fn f() {
    unsafe {
        foo();
    }
}
