#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use abi::device::{PCI_IRQ_MODE_MSI, PCI_IRQ_MODE_MSIX};
use abi::errors::Errno;
use core::mem::size_of;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};
use stem::device::device_enable_msi;
use stem::syscall::{device_alloc_dma, device_claim, device_dma_phys, device_map_mmio};
use stem::{info, warn};

const RTL8168_VENDOR_ID: u16 = 0x10ec;
const RTL8168_DEVICE_ID: u16 = 0x8168;

const RX_RING_SIZE: usize = 64;
const TX_RING_SIZE: usize = 16;
const RX_BUF_SIZE: usize = 2048;
const TX_BUF_SIZE: usize = 2048;

const DESC_OWN: u32 = 1 << 31;
const DESC_EOR: u32 = 1 << 30;
const DESC_FS: u32 = 1 << 29;
const DESC_LS: u32 = 1 << 28;

const REG_IDR0: u32 = 0x00;
const REG_COMMAND: u32 = 0x37;
const REG_TPPOLL: u32 = 0x38;
const REG_IMR: u32 = 0x3c;
const REG_ISR: u32 = 0x3e;
const REG_TCR: u32 = 0x40;
const REG_RCR: u32 = 0x44;
const REG_PHYSTATUS: u32 = 0x6c;
const REG_CPLUSCMD: u32 = 0xe0;
const REG_TX_DESC_LOW: u32 = 0x20;
const REG_TX_DESC_HIGH: u32 = 0x24;
const REG_RX_DESC_LOW: u32 = 0xe4;
const REG_RX_DESC_HIGH: u32 = 0xe8;

const CMD_RX_ENABLE: u8 = 1 << 3;
const CMD_TX_ENABLE: u8 = 1 << 2;
const CMD_RESET: u8 = 1 << 4;

const TPPOLL_NPQ: u8 = 0x40;
const PHYSTATUS_LINK: u8 = 1 << 1;
const ISR_RX_OK: u16 = 1 << 0;
const ISR_RX_ERR: u16 = 1 << 1;
const ISR_TX_OK: u16 = 1 << 2;
const ISR_TX_ERR: u16 = 1 << 3;
const ISR_RX_OVERFLOW: u16 = 1 << 4;
const ISR_LINK_CHG: u16 = 1 << 5;
const ISR_RX_FIFO_OVER: u16 = 1 << 6;
const ISR_SYS_ERR: u16 = 1 << 15;
const IRQ_MASK: u16 = ISR_RX_OK
    | ISR_RX_ERR
    | ISR_TX_OK
    | ISR_TX_ERR
    | ISR_RX_OVERFLOW
    | ISR_LINK_CHG
    | ISR_RX_FIFO_OVER
    | ISR_SYS_ERR;

#[repr(C, align(16))]
#[derive(Copy, Clone, Default)]
struct DmaDesc {
    opts1: u32,
    opts2: u32,
    addr_lo: u32,
    addr_hi: u32,
}

pub struct Rtl8168Driver {
    mmio: u64,
    mac: [u8; 6],
    link_up: bool,
    claim: usize,
    irq_enabled: bool,

    rx_desc_virt: u64,
    rx_desc_phys: u64,
    tx_desc_virt: u64,
    tx_desc_phys: u64,

    rx_buf_virt: [u64; RX_RING_SIZE],
    rx_buf_phys: [u64; RX_RING_SIZE],
    tx_buf_virt: [u64; TX_RING_SIZE],
    tx_buf_phys: [u64; TX_RING_SIZE],

    rx_idx: usize,
    tx_idx: usize,
    rx_scratch: [u8; RX_BUF_SIZE],
}

impl Rtl8168Driver {
    pub fn new(sysfs_path: &str) -> Result<Self, Errno> {
        // Claim the device using its sysfs path as the primary key.
        info!("RTL8168: claiming device at {}", sysfs_path);

        let claim = device_claim(sysfs_path)?;
        let mmio = device_map_mmio(claim, 0)?;
        info!("RTL8168: BAR0 mapped at 0x{:x}", mmio);

        let rx_desc_virt = device_alloc_dma(claim, 1)?;
        let rx_desc_phys = device_dma_phys(rx_desc_virt)?;
        let tx_desc_virt = device_alloc_dma(claim, 1)?;
        let tx_desc_phys = device_dma_phys(tx_desc_virt)?;

        let mut rx_buf_virt = [0u64; RX_RING_SIZE];
        let mut rx_buf_phys = [0u64; RX_RING_SIZE];
        for i in 0..RX_RING_SIZE {
            rx_buf_virt[i] = device_alloc_dma(claim, 1)?;
            rx_buf_phys[i] = device_dma_phys(rx_buf_virt[i])?;
        }

        let mut tx_buf_virt = [0u64; TX_RING_SIZE];
        let mut tx_buf_phys = [0u64; TX_RING_SIZE];
        for i in 0..TX_RING_SIZE {
            tx_buf_virt[i] = device_alloc_dma(claim, 1)?;
            tx_buf_phys[i] = device_dma_phys(tx_buf_virt[i])?;
        }

        let mut driver = Self {
            mmio,
            mac: [0; 6],
            link_up: false,
            claim,
            irq_enabled: false,
            rx_desc_virt,
            rx_desc_phys,
            tx_desc_virt,
            tx_desc_phys,
            rx_buf_virt,
            rx_buf_phys,
            tx_buf_virt,
            tx_buf_phys,
            rx_idx: 0,
            tx_idx: 0,
            rx_scratch: [0; RX_BUF_SIZE],
        };

        driver.reset_chip();
        driver.init_rings();
        driver.configure_chip();
        driver.try_enable_irq();
        driver.mac = driver.read_mac();
        driver.link_up = driver.read_link_up();

        info!(
            "RTL8168: init done, MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, link={}",
            driver.mac[0],
            driver.mac[1],
            driver.mac[2],
            driver.mac[3],
            driver.mac[4],
            driver.mac[5],
            if driver.link_up { "up" } else { "down" }
        );

        Ok(driver)
    }

    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    pub fn link_up(&self) -> bool {
        self.read_link_up()
    }

    pub fn irq_enabled(&self) -> bool {
        self.irq_enabled
    }

    pub fn wait_for_irq(&self) {
        if !self.irq_enabled {
            stem::time::sleep_ms(1);
            return;
        }
        let _ = stem::syscall::device_irq_wait(self.claim, 0);
    }

    pub fn consume_interrupts(&mut self) -> u16 {
        let isr = self.read_u16(REG_ISR);
        if isr != 0 {
            self.write_u16(REG_ISR, isr);
        }
        if (isr & ISR_LINK_CHG) != 0 {
            self.link_up = self.read_link_up();
        }
        isr
    }

    pub fn poll_rx(&mut self) -> Option<&[u8]> {
        let idx = self.rx_idx;
        let opts1 = unsafe { (*self.rx_desc_ptr(idx)).opts1 };
        if (opts1 & DESC_OWN) != 0 {
            return None;
        }

        let len = (opts1 & 0x3fff) as usize;
        if len >= 4 && len <= RX_BUF_SIZE {
            let frame_len = len - 4; // strip CRC
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.rx_buf_virt[idx] as *const u8,
                    self.rx_scratch.as_mut_ptr(),
                    frame_len,
                );
            }

            let mut new_opts1 = DESC_OWN | (RX_BUF_SIZE as u32);
            if idx == RX_RING_SIZE - 1 {
                new_opts1 |= DESC_EOR;
            }
            unsafe {
                (*self.rx_desc_ptr(idx)).opts1 = new_opts1;
            }
            fence(Ordering::Release);

            self.rx_idx = (idx + 1) % RX_RING_SIZE;
            return Some(&self.rx_scratch[..frame_len]);
        }

        let mut new_opts1 = DESC_OWN | (RX_BUF_SIZE as u32);
        if idx == RX_RING_SIZE - 1 {
            new_opts1 |= DESC_EOR;
        }
        unsafe {
            (*self.rx_desc_ptr(idx)).opts1 = new_opts1;
        }
        self.rx_idx = (idx + 1) % RX_RING_SIZE;
        None
    }

    pub fn tx(&mut self, frame: &[u8]) -> Result<(), &'static str> {
        if frame.len() > TX_BUF_SIZE {
            return Err("frame too large");
        }
        let idx = self.tx_idx;
        if unsafe { (*self.tx_desc_ptr(idx)).opts1 & DESC_OWN } != 0 {
            return Err("tx ring full");
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                frame.as_ptr(),
                self.tx_buf_virt[idx] as *mut u8,
                frame.len(),
            );
        }

        fence(Ordering::Release);
        let mut opts1 = DESC_OWN | DESC_FS | DESC_LS | (frame.len() as u32);
        if idx == TX_RING_SIZE - 1 {
            opts1 |= DESC_EOR;
        }
        unsafe {
            let desc = self.tx_desc_ptr(idx);
            (*desc).opts1 = opts1;
            (*desc).opts2 = 0;
        }

        self.write_u8(REG_TPPOLL, TPPOLL_NPQ);
        self.tx_idx = (idx + 1) % TX_RING_SIZE;
        Ok(())
    }

    fn init_rings(&mut self) {
        for i in 0..RX_RING_SIZE {
            unsafe {
                let desc = self.rx_desc_ptr(i);
                (*desc).addr_lo = self.rx_buf_phys[i] as u32;
                (*desc).addr_hi = (self.rx_buf_phys[i] >> 32) as u32;
                (*desc).opts2 = 0;
                (*desc).opts1 = DESC_OWN
                    | (RX_BUF_SIZE as u32)
                    | if i == RX_RING_SIZE - 1 { DESC_EOR } else { 0 };
            }
        }
        for i in 0..TX_RING_SIZE {
            unsafe {
                let desc = self.tx_desc_ptr(i);
                (*desc).addr_lo = self.tx_buf_phys[i] as u32;
                (*desc).addr_hi = (self.tx_buf_phys[i] >> 32) as u32;
                (*desc).opts1 = if i == TX_RING_SIZE - 1 { DESC_EOR } else { 0 };
                (*desc).opts2 = 0;
            }
        }
        fence(Ordering::SeqCst);
    }

    fn configure_chip(&mut self) {
        self.write_u16(REG_IMR, 0); // Keep masked until IRQ path is configured.
        self.write_u16(REG_ISR, 0xffff);

        self.write_u32(REG_TX_DESC_LOW, self.tx_desc_phys as u32);
        self.write_u32(REG_TX_DESC_HIGH, (self.tx_desc_phys >> 32) as u32);
        self.write_u32(REG_RX_DESC_LOW, self.rx_desc_phys as u32);
        self.write_u32(REG_RX_DESC_HIGH, (self.rx_desc_phys >> 32) as u32);

        // Conservative defaults used by many RTL8169/8168 bare-metal bringups.
        self.write_u16(REG_CPLUSCMD, 0);
        self.write_u32(REG_TCR, 0x0300_0700);
        self.write_u32(REG_RCR, 0x0000_e70e);

        let cmd = CMD_RX_ENABLE | CMD_TX_ENABLE;
        self.write_u8(REG_COMMAND, cmd);
        self.write_u16(REG_ISR, 0xffff);
    }

    fn try_enable_irq(&mut self) {
        match device_enable_msi(self.claim, true) {
            Ok(resp) => {
                let mode = match resp.irq_mode {
                    PCI_IRQ_MODE_MSI => "msi",
                    PCI_IRQ_MODE_MSIX => "msix",
                    _ => "unknown",
                };
                match stem::syscall::device_irq_subscribe(self.claim, 0) {
                    Ok(_) => {
                        self.irq_enabled = true;
                        self.write_u16(REG_ISR, 0xffff);
                        self.write_u16(REG_IMR, IRQ_MASK);
                        info!(
                            "RTL8168: IRQ enabled via {} vector=0x{:02x}",
                            mode, resp.vector
                        );
                    }
                    Err(e) => {
                        warn!(
                            "RTL8168: irq subscribe failed after {} setup: {:?}",
                            mode, e
                        );
                        self.irq_enabled = false;
                        self.write_u16(REG_IMR, 0);
                    }
                }
            }
            Err(e) => {
                warn!(
                    "RTL8168: MSI/MSI-X unavailable, using polling mode: {:?}",
                    e
                );
                self.irq_enabled = false;
                self.write_u16(REG_IMR, 0);
            }
        }
    }

    fn reset_chip(&self) {
        self.write_u8(REG_COMMAND, CMD_RESET);
        for _ in 0..100_000 {
            if (self.read_u8(REG_COMMAND) & CMD_RESET) == 0 {
                return;
            }
            core::hint::spin_loop();
        }
        warn!("RTL8168: reset timeout");
    }

    fn read_mac(&self) -> [u8; 6] {
        [
            self.read_u8(REG_IDR0),
            self.read_u8(REG_IDR0 + 1),
            self.read_u8(REG_IDR0 + 2),
            self.read_u8(REG_IDR0 + 3),
            self.read_u8(REG_IDR0 + 4),
            self.read_u8(REG_IDR0 + 5),
        ]
    }

    fn read_link_up(&self) -> bool {
        (self.read_u8(REG_PHYSTATUS) & PHYSTATUS_LINK) != 0
    }

    fn rx_desc_ptr(&self, idx: usize) -> *mut DmaDesc {
        debug_assert!(idx < RX_RING_SIZE);
        unsafe { (self.rx_desc_virt as *mut DmaDesc).add(idx) }
    }

    fn tx_desc_ptr(&self, idx: usize) -> *mut DmaDesc {
        debug_assert!(idx < TX_RING_SIZE);
        unsafe { (self.tx_desc_virt as *mut DmaDesc).add(idx) }
    }

    fn read_u8(&self, reg: u32) -> u8 {
        unsafe { read_volatile((self.mmio + reg as u64) as *const u8) }
    }
    fn read_u16(&self, reg: u32) -> u16 {
        unsafe { read_volatile((self.mmio + reg as u64) as *const u16) }
    }
    fn write_u8(&self, reg: u32, value: u8) {
        unsafe { write_volatile((self.mmio + reg as u64) as *mut u8, value) }
    }
    fn write_u16(&self, reg: u32, value: u16) {
        unsafe { write_volatile((self.mmio + reg as u64) as *mut u16, value) }
    }
    fn write_u32(&self, reg: u32, value: u32) {
        unsafe { write_volatile((self.mmio + reg as u64) as *mut u32, value) }
    }
}

const _: () = {
    assert!(size_of::<DmaDesc>() == 16);
};
