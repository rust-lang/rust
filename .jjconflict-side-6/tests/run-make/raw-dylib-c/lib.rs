#[link(name = "extern_1.dll", kind = "raw-dylib", modifiers = "+verbatim")]
extern "C" {
    fn extern_fn_1();
}

#[link(name = "extern_2", kind = "raw-dylib")]
extern "C" {
    fn extern_fn_3();
}

pub fn library_function() {
    #[link(name = "extern_1", kind = "raw-dylib")]
    extern "C" {
        fn extern_fn_2();
        fn print_extern_variable();
        static mut extern_variable: i32;
        #[link_name = "extern_fn_4"]
        fn extern_fn_4_renamed();
    }

    unsafe {
        extern_fn_1();
        extern_fn_2();
        extern_fn_3();
        extern_fn_4_renamed();
        extern_variable = 42;
        print_extern_variable();
        extern_variable = -42;
        print_extern_variable();
    }
}

fn main() {
    library_function();
}
