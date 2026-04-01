#![feature(abi_vectorcall)]

#[link(name = "extern", kind = "raw-dylib", import_name_type = "undecorated")]
extern "C" {
    fn LooksLikeAPrivateGlobal(i: i32);
    fn cdecl_fn_undecorated(i: i32);
    #[link_name = "cdecl_fn_undecorated2"]
    fn cdecl_fn_undecorated_renamed(i: i32);
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
    #[link_name = "stdcall_fn_undecorated2"]
    fn stdcall_fn_undecorated_renamed(i: i32);
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
    #[link_name = "fastcall_fn_undecorated2"]
    fn fastcall_fn_undecorated_renamed(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "noprefix")]
extern "fastcall" {
    fn fastcall_fn_noprefix(i: i32);
}

#[link(name = "extern", kind = "raw-dylib", import_name_type = "decorated")]
extern "fastcall" {
    fn fastcall_fn_decorated(i: i32);
}

#[cfg(target_env = "msvc")]
#[link(name = "extern", kind = "raw-dylib", import_name_type = "undecorated")]
extern "vectorcall" {
    fn vectorcall_fn_undecorated(i: i32);
    #[link_name = "vectorcall_fn_undecorated2"]
    fn vectorcall_fn_undecorated_renamed(i: i32);
}

#[cfg(target_env = "msvc")]
#[link(name = "extern", kind = "raw-dylib", import_name_type = "noprefix")]
extern "vectorcall" {
    fn vectorcall_fn_noprefix(i: i32);
}

#[cfg(target_env = "msvc")]
#[link(name = "extern", kind = "raw-dylib", import_name_type = "decorated")]
extern "vectorcall" {
    fn vectorcall_fn_decorated(i: i32);
}

#[link(name = "extern", kind = "raw-dylib")]
extern "C" {
    fn print_extern_variable_undecorated();
    fn print_extern_variable_noprefix();
    fn print_extern_variable_decorated();
}

pub fn main() {
    unsafe {
        // Regression test for #104453
        // On x86 LLVM uses 'L' as the prefix for private globals (PrivateGlobalPrefix), which
        // causes it to believe that undecorated functions starting with 'L' are actually temporary
        // symbols that it generated, which causes a later check to fail as the symbols we are
        // creating don't have definitions (whereas all temporary symbols do).
        LooksLikeAPrivateGlobal(13);

        cdecl_fn_undecorated(1);
        cdecl_fn_undecorated_renamed(10);
        cdecl_fn_noprefix(2);
        cdecl_fn_decorated(3);

        stdcall_fn_undecorated(4);
        stdcall_fn_undecorated_renamed(14);
        stdcall_fn_noprefix(5);
        stdcall_fn_decorated(6);

        fastcall_fn_undecorated(7);
        fastcall_fn_undecorated_renamed(17);
        fastcall_fn_noprefix(8);
        fastcall_fn_decorated(9);

        extern_variable_undecorated = 42;
        print_extern_variable_undecorated();
        extern_variable_noprefix = 43;
        print_extern_variable_noprefix();
        extern_variable_decorated = 44;
        print_extern_variable_decorated();

        // GCC doesn't support vectorcall: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89485
        #[cfg(target_env = "msvc")]
        {
            vectorcall_fn_undecorated(10);
            vectorcall_fn_undecorated_renamed(20);
            vectorcall_fn_noprefix(11);
            vectorcall_fn_decorated(12);
        }
        #[cfg(not(target_env = "msvc"))]
        {
            println!("vectorcall_fn_undecorated(10)");
            println!("vectorcall_fn_undecorated2(20)");
            println!("vectorcall_fn_noprefix(11)");
            println!("vectorcall_fn_decorated(12)");
        }
    }
}
