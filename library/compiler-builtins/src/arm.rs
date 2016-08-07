use core::intrinsics;

macro_rules! defer {
    ($($symbol:ident),+ -> $routine:ident) => {
        $(
            #[naked]
            #[no_mangle]
            pub extern "C" fn $symbol() {
                unsafe {
                    asm!(concat!("b ", stringify!($routine)));
                    intrinsics::unreachable();
                }
            }
        )+
    }
}

// FIXME only `__aeabi_memcmp` should be defined like this. The `*4` and `*8` variants should be
// defined as aliases of `__aeabi_memcmp`
defer!(__aeabi_memcmp, __aeabi_memcmp4, __aeabi_memcmp8 -> memcmp);

// FIXME same issue as `__aeabi_memcmp*`
defer!(__aeabi_memcpy, __aeabi_memcpy4, __aeabi_memcpy8 -> memcpy);

// FIXME same issue as `__aeabi_memcmp*`
defer!(__aeabi_memmove, __aeabi_memmove4, __aeabi_memmove8 -> memmove);

macro_rules! memset {
    ($($symbol:ident),+) => {
        $(
            #[naked]
            #[no_mangle]
            pub extern "C" fn $symbol() {
                unsafe {
                    asm!("mov r3, r1
                          mov r1, r2
                          mov r2, r3
                          b memset");
                    intrinsics::unreachable();
                }
            }
        )+
    }
}

// FIXME same issue as `__aeabi_memcmp*`
memset!(__aeabi_memset, __aeabi_memset4, __aeabi_memset8);

macro_rules! memclr {
    ($($symbol:ident),+) => {
        $(
            #[naked]
            #[no_mangle]
            pub extern "C" fn $symbol() {
                unsafe {
                    asm!("mov r2, r1
                          mov r1, #0
                          b memset");
                    intrinsics::unreachable();
                }
            }
        )+
    }
}

// FIXME same issue as `__aeabi_memcmp*`
memclr!(__aeabi_memclr, __aeabi_memclr4, __aeabi_memclr8);
