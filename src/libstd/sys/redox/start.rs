use sys::syscall::exit;

#[allow(private_no_mangle_fns)]
#[no_mangle]
#[naked]
#[cfg(target_arch = "x86")]
pub unsafe fn _start() {
    asm!("push esp
        call _start_stack
        pop esp"
        :
        :
        : "memory"
        : "intel", "volatile");
    let _ = exit(0);
}

#[allow(private_no_mangle_fns)]
#[no_mangle]
#[naked]
#[cfg(target_arch = "x86_64")]
pub unsafe fn _start() {
    asm!("mov rdi, rsp
        and rsp, 0xFFFFFFFFFFFFFFF0
        call _start_stack"
        :
        :
        : "memory"
        : "intel", "volatile");
    let _ = exit(0);
}

#[allow(private_no_mangle_fns)]
#[no_mangle]
pub unsafe extern "C" fn _start_stack(stack: *const usize){
    extern "C" {
        fn main(argc: usize, argv: *const *const u8) -> usize;
    }

    let argc = *stack as usize;
    let argv = stack.offset(1) as *const *const u8;
    let _ = exit(main(argc, argv));
}
