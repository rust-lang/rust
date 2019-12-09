use super::mem;
use crate::slice::from_raw_parts;

const R_X86_64_RELATIVE: u32 = 8;

#[repr(packed)]
struct Rela<T> {
    offset: T,
    info: T,
    addend: T,
}

pub fn relocate_elf_rela() {
    extern "C" {
        static RELA: u64;
        static RELACOUNT: usize;
    }

    if unsafe { RELACOUNT } == 0 {
        return;
    } // unsafe ok: link-time constant

    let relas = unsafe {
        from_raw_parts::<Rela<u64>>(mem::rel_ptr(RELA), RELACOUNT) // unsafe ok: link-time constant
    };
    for rela in relas {
        if rela.info != (/*0 << 32 |*/R_X86_64_RELATIVE as u64) {
            rtabort!("Invalid relocation");
        }
        unsafe { *mem::rel_ptr_mut::<*const ()>(rela.offset) = mem::rel_ptr(rela.addend) };
    }
}
