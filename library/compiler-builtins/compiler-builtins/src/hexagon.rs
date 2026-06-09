use core::arch::global_asm;

// Hexagon L1 cache line size in bytes (Hexagon PRM sections 5.10.3-5.10.4).
const CACHE_LINE_SIZE: usize = 32;

intrinsics! {
    pub unsafe extern "C" fn __clear_cache(start: *mut u8, end: *mut u8) {
        // Hexagon has separate instruction and data caches.
        let mask = !(CACHE_LINE_SIZE - 1);
        let start_line = start.addr() & mask;
        let end_addr = end.addr();

        // Clean and invalidate data cache to push new code to memory and
        // invalidate stale lines in the L2 cache.
        let mut addr = start_line;
        while addr < end_addr {
            unsafe {
                core::arch::asm!(
                    "dccleaninva({addr})",
                    addr = in(reg) addr,
                    options(nostack, preserves_flags),
                );
            }
            addr += CACHE_LINE_SIZE;
        }

        // Invalidate instruction cache so it re-fetches from memory.
        addr = start_line;
        while addr < end_addr {
            unsafe {
                core::arch::asm!(
                    "icinva({addr})",
                    addr = in(reg) addr,
                    options(nostack, preserves_flags),
                );
            }
            addr += CACHE_LINE_SIZE;
        }

        // Instruction sync barrier ensures subsequent fetches see the new code.
        unsafe {
            core::arch::asm!("isync", options(nostack, preserves_flags));
        }
    }
}

global_asm!(include_str!("hexagon/func_macro.s"), options(raw));

global_asm!(
    include_str!("hexagon/common_entry_exit_abi1.s"),
    options(raw)
);

global_asm!(
    include_str!("hexagon/common_entry_exit_abi2.s"),
    options(raw)
);

global_asm!(
    include_str!("hexagon/common_entry_exit_legacy.s"),
    options(raw)
);

global_asm!(include_str!("hexagon/dfaddsub.s"), options(raw));

global_asm!(include_str!("hexagon/dfdiv.s"), options(raw));

global_asm!(include_str!("hexagon/dffma.s"), options(raw));

global_asm!(include_str!("hexagon/dfminmax.s"), options(raw));

global_asm!(include_str!("hexagon/dfmul.s"), options(raw));

global_asm!(include_str!("hexagon/dfsqrt.s"), options(raw));

global_asm!(include_str!("hexagon/divdi3.s"), options(raw));

global_asm!(include_str!("hexagon/divsi3.s"), options(raw));

global_asm!(include_str!("hexagon/fastmath2_dlib_asm.s"), options(raw));

global_asm!(include_str!("hexagon/fastmath2_ldlib_asm.s"), options(raw));

global_asm!(
    include_str!("hexagon/memcpy_forward_vp4cp4n2.s"),
    options(raw)
);

global_asm!(
    include_str!("hexagon/memcpy_likely_aligned.s"),
    options(raw)
);

global_asm!(include_str!("hexagon/moddi3.s"), options(raw));

global_asm!(include_str!("hexagon/modsi3.s"), options(raw));

global_asm!(include_str!("hexagon/sfdiv_opt.s"), options(raw));

global_asm!(include_str!("hexagon/sfsqrt_opt.s"), options(raw));

global_asm!(include_str!("hexagon/udivdi3.s"), options(raw));

global_asm!(include_str!("hexagon/udivmoddi4.s"), options(raw));

global_asm!(include_str!("hexagon/udivmodsi4.s"), options(raw));

global_asm!(include_str!("hexagon/udivsi3.s"), options(raw));

global_asm!(include_str!("hexagon/umoddi3.s"), options(raw));

global_asm!(include_str!("hexagon/umodsi3.s"), options(raw));
