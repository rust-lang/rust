#![crate_type = "rlib"]
#![crate_type = "cdylib"]

#[macro_export]
macro_rules! asm_func {
    ($name:expr, $body:expr $(, $($args:tt)*)?) => {
        core::arch::global_asm!(
            concat!(
                ".p2align 4\n",
                ".hidden ", $name, "\n",
                ".global ", $name, "\n",
                ".type ", $name, ",@function\n",
                $name, ":\n",
                $body,
                ".size ", $name, ",.-", $name,
            )
            $(, $($args)*)?
        );
    };
}

macro_rules! libcall_trampoline {
    ($libcall:ident ; $libcall_impl:ident) => {
        asm_func!(
            stringify!($libcall),
            concat!(
                "
                   .cfi_startproc simple
                   .cfi_def_cfa_offset 0
                    jmp {}
                    .cfi_endproc
                ",
            ),
            sym $libcall_impl
        );
    };
}

pub mod trampolines {
    extern "C" {
        pub fn table_fill_funcref();
        pub fn table_fill_externref();
    }

    unsafe extern "C" fn impl_table_fill_funcref() {}
    unsafe extern "C" fn impl_table_fill_externref() {}

    libcall_trampoline!(table_fill_funcref ; impl_table_fill_funcref);
    libcall_trampoline!(table_fill_externref ; impl_table_fill_externref);
}
