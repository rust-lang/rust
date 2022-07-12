#![feature(raw_dylib)]

#[link(name = "extern", kind = "raw-dylib", import_name_type = "undecorated")]
extern "C" {
    fn cdecl_fn_undecorated(i: i32);
    static mut extern_variable_undecorated: i32;
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "noprefix")]
extern "C" {
    fn cdecl_fn_noprefix(i: i32);
    static mut extern_variable_noprefix: i32;
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "decorated")]
extern "C" {
    fn cdecl_fn_decorated(i: i32);
    static mut extern_variable_decorated: i32;
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "undecorated")]
extern "stdcall" {
    fn stdcall_fn_undecorated(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "noprefix")]
extern "stdcall" {
    fn stdcall_fn_noprefix(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "decorated")]
extern "stdcall" {
    fn stdcall_fn_decorated(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "undecorated")]
extern "fastcall" {
    fn fastcall_fn_undecorated(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "noprefix")]
extern "fastcall" {
    fn fastcall_fn_noprefix(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "decorated")]
extern "fastcall" {
    fn fastcall_fn_decorated(i: i32);
}

#[link(name = "extern", kind = "raw-dylib")]
extern {
    fn print_extern_variable_undecorated();
    fn print_extern_variable_noprefix();
    fn print_extern_variable_decorated();
}

pub fn main() {
    unsafe {
        cdecl_fn_undecorated(1);
        cdecl_fn_noprefix(2);
        cdecl_fn_decorated(3);

        stdcall_fn_undecorated(4);
        stdcall_fn_noprefix(5);
        stdcall_fn_decorated(6);

        fastcall_fn_undecorated(7);
        fastcall_fn_noprefix(8);
        fastcall_fn_decorated(9);

        extern_variable_undecorated = 42;
        print_extern_variable_undecorated();
        extern_variable_noprefix = 43;
        print_extern_variable_noprefix();
        extern_variable_decorated = 44;
        print_extern_variable_decorated();
    }
}
