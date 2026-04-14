// x86_64 PCI Legacy Configuration Space Access

use spin::Mutex;

const PCI_CONFIG_ADDRESS: u16 = 0xCF8;
const PCI_CONFIG_DATA: u16 = 0xCFC;
const PCI_ENABLE_BIT: u32 = 0x80000000;

static PCI_LOCK: Mutex<()> = Mutex::new(());

#[inline]
unsafe fn outl(port: u16, val: u32) {
    unsafe {
        core::arch::asm!("out dx, eax", in("dx") port, in("eax") val, options(nostack, preserves_flags));
    }
}

#[inline]
unsafe fn inl(port: u16) -> u32 {
    let ret: u32;
    unsafe {
        core::arch::asm!("in eax, dx", out("eax") ret, in("dx") port, options(nostack, preserves_flags));
    }
    ret
}

pub fn read_config(bus: u8, dev: u8, func: u8, offset: u8) -> u32 {
    let address = PCI_ENABLE_BIT
        | ((bus as u32) << 16)
        | ((dev as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xFC);

    let _lock = PCI_LOCK.lock();
    unsafe {
        outl(PCI_CONFIG_ADDRESS, address);
        inl(PCI_CONFIG_DATA)
    }
}

pub fn write_config(bus: u8, dev: u8, func: u8, offset: u8, value: u32) {
    let address = PCI_ENABLE_BIT
        | ((bus as u32) << 16)
        | ((dev as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xFC);

    let _lock = PCI_LOCK.lock();
    unsafe {
        outl(PCI_CONFIG_ADDRESS, address);
        outl(PCI_CONFIG_DATA, value);
    }
}
