use sys::syscall::exit;

#[unstable(feature = "start_fn", issue = "0")]
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

#[unstable(feature = "start_fn", issue = "0")]
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

#[unstable(feature = "start_fn", issue = "0")]
#[no_mangle]
pub unsafe extern "C" fn _start_stack(stack: *const usize){
    extern "C" {
        fn main(argc: usize, argv: *const *const u8) -> usize;
    }

    let argc = *stack as usize;
    let argv = stack.offset(1) as *const *const u8;
    let _ = exit(main(argc, argv));
}

/// Memcpy
///
/// Copy N bytes of memory from one location to another.
#[unstable(feature = "start_fn", issue = "0")]
#[no_mangle]
pub unsafe extern fn memcpy(dest: *mut u8, src: *const u8,
                            n: usize) -> *mut u8 {
    let mut i = 0;
    while i < n {
        *((dest as usize + i) as *mut u8) = *((src as usize + i) as *const u8);
        i += 1;
    }

    dest
}

/// Memmove
///
/// Copy N bytes of memory from src to dest. The memory areas may overlap.
#[unstable(feature = "start_fn", issue = "0")]
#[no_mangle]
pub unsafe extern fn memmove(dest: *mut u8, src: *const u8,
                             n: usize) -> *mut u8 {
    if src < dest as *const u8 {
        let mut i = n;
        while i != 0 {
            i -= 1;
            *((dest as usize + i) as *mut u8) = *((src as usize + i) as *const u8);
        }
    } else {
        let mut i = 0;
        while i < n {
            *((dest as usize + i) as *mut u8) = *((src as usize + i) as *const u8);
            i += 1;
        }
    }

    dest
}

/// Memset
///
/// Fill a block of memory with a specified value.
#[unstable(feature = "start_fn", issue = "0")]
#[no_mangle]
pub unsafe extern fn memset(dest: *mut u8, c: i32, n: usize) -> *mut u8 {
    let mut i = 0;
    while i < n {
        *((dest as usize + i) as *mut u8) = c as u8;
        i += 1;
    }

    dest
}

/// Memcmp
///
/// Compare two blocks of memory.
#[unstable(feature = "start_fn", issue = "0")]
#[no_mangle]
pub unsafe extern fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
    let mut i = 0;

    while i < n {
        let a = *((s1 as usize + i) as *const u8);
        let b = *((s2 as usize + i) as *const u8);
        if a != b {
            return a as i32 - b as i32
        }
        i += 1;
    }

    0
}
