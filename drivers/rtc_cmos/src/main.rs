#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::abi::driver_ctx::DriverCtx;
use stem::abi::module_manifest::{ManifestHeader, ModuleKind, MANIFEST_MAGIC};
use stem::{debug, error, info, warn};

#[link_section = ".thing_manifest"]
#[no_mangle]
#[used]
pub static MANIFEST: ManifestHeader = ManifestHeader {
    magic: MANIFEST_MAGIC,
    kind: ModuleKind::Driver,
    device_kind: [
        0x64, 0x65, 0x76, 0x2e, 0x72, 0x74, 0x63, 0x2e, 0x43, 0x6d, 0x6f, 0x73, 0x00, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    version: 1,
    _reserved: 0,
};

// CMOS RTC I/O ports
const CMOS_ADDR: u16 = 0x70;
const CMOS_DATA: u16 = 0x71;

// CMOS registers
const RTC_SECONDS: u8 = 0x00;
const RTC_MINUTES: u8 = 0x02;
const RTC_HOURS: u8 = 0x04;
const RTC_DAY: u8 = 0x07;
const RTC_MONTH: u8 = 0x08;
const RTC_YEAR: u8 = 0x09;
const RTC_STATUS_A: u8 = 0x0A;
const RTC_STATUS_B: u8 = 0x0B;

fn cmos_read(reg: u8) -> u8 {
    stem::syscall::ioport_write(CMOS_ADDR as usize, reg as usize, 1);
    stem::syscall::ioport_read(CMOS_DATA as usize, 1) as u8
}

fn is_updating() -> bool {
    cmos_read(RTC_STATUS_A) & 0x80 != 0
}

fn bcd_to_binary(bcd: u8) -> u8 {
    (bcd & 0x0F) + ((bcd >> 4) * 10)
}

fn read_rtc() -> (u16, u8, u8, u8, u8, u8) {
    // Wait for update to complete
    while is_updating() {
        stem::yield_now();
    }

    let seconds = cmos_read(RTC_SECONDS);
    let minutes = cmos_read(RTC_MINUTES);
    let hours = cmos_read(RTC_HOURS);
    let day = cmos_read(RTC_DAY);
    let month = cmos_read(RTC_MONTH);
    let year = cmos_read(RTC_YEAR);

    let status_b = cmos_read(RTC_STATUS_B);
    let is_bcd = (status_b & 0x04) == 0;

    let (s, m, h, d, mo, y) = if is_bcd {
        (
            bcd_to_binary(seconds),
            bcd_to_binary(minutes),
            bcd_to_binary(hours & 0x7F), // Mask out PM bit for 12-hour mode
            bcd_to_binary(day),
            bcd_to_binary(month),
            bcd_to_binary(year),
        )
    } else {
        (seconds, minutes, hours & 0x7F, day, month, year)
    };

    // Assume century is 20xx for year < 70, 19xx for >= 70
    let full_year = if y < 70 {
        2000 + y as u16
    } else {
        1900 + y as u16
    };

    (full_year, mo, d, h, m, s)
}

/// Convert calendar time to Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)
fn rtc_to_unix(year: u16, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> u64 {
    const MONTH_DAYS: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

    let year = year as u64;
    let month = (month as u64).clamp(1, 12);
    let day = (day as u64).clamp(1, 31);

    let years = year.saturating_sub(1970);

    // Count leap years since 1970
    let leap_years = if year > 1970 {
        let y = year - 1;
        let from_base = |y: u64| (y / 4) - (y / 100) + (y / 400);
        from_base(y) - from_base(1969)
    } else {
        0
    };

    let mut days = years * 365 + leap_years;
    days += MONTH_DAYS[(month - 1) as usize] as u64;

    let is_leap = (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
    if is_leap && month > 2 {
        days += 1;
    }

    days += day - 1;

    days * 86400 + hour as u64 * 3600 + minute as u64 * 60 + second as u64
}

#[stem::main]
fn main(arg: usize) -> ! {
    let cpu = stem::arch::whoami();
    debug!(
        "whoami: cs=0x{:x} ss=0x{:x} cpl={} rsp=0x{:x} rip=0x{:x} rflags=0x{:x}",
        cpu.cs, cpu.ss, cpu.cpl, cpu.rsp, cpu.rip, cpu.rflags
    );

    debug!("Starting... arg={:x}", arg);

    let dev_id = if arg != 0 {
        let ctx = DriverCtx::from_raw(arg);
        let id = stem::thing::ThingId(ctx.device_id.0);
        debug!("Serving device ID: {:?}", id);
        id
    } else {
        debug!("Starting without explicit context (phased boot mode).");
        Default::default()
    };

    // Read RTC and anchor system clock
    let (year, month, day, hour, minute, second) = read_rtc();
    let unix_secs = rtc_to_unix(year, month, day, hour, minute, second);

    debug!(
        "RTC: {:04}-{:02}-{:02} {:02}:{:02}:{:02} = {} unix_secs",
        year, month, day, hour, minute, second, unix_secs
    );

    // Anchor the system clock!
    stem::syscall::time_anchor(unix_secs);
    debug!("RTC: System clock anchored to {} unix_secs", unix_secs);

    debug!("RTC: Entering maintenance loop.");

    loop {
        stem::sleep(core::time::Duration::from_secs(3600)); // Update once per hour
        let (year, month, day, hour, minute, second) = read_rtc();
        let unix_secs = rtc_to_unix(year, month, day, hour, minute, second);
        stem::syscall::time_anchor(unix_secs);
    }
}
