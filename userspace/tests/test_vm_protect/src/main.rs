#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use abi::vm::{VmMapFlags, VmProt};
use stem::println;
use stem::vm::*;

#[stem::main]
fn main(_arg0: usize) -> ! {
    println!("--- test_vm_protect starting ---");

    // 1. Map an anonymous RW region
    let map_req = VmMapReq {
        addr_hint: 0x4000_0000,
        len: 0x3000, // 3 pages
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::FIXED | VmMapFlags::PRIVATE,
        backing: abi::vm::VmBacking::Anonymous { zeroed: true },
    };
    let map_resp = vm_map(&map_req).expect("vm_map failed");
    let ptr = map_resp.addr as *mut u8;
    println!("Mapped RW at 0x{:x}", map_resp.addr);

    // 2. Write to it to ensure it's writable
    unsafe {
        ptr.write_volatile(0x42);
        ptr.add(0x1000).write_volatile(0x43);
        ptr.add(0x2000).write_volatile(0x44);
    }
    println!("Verified writable");

    // 3. Downgrade middle page to RO
    let prot_req = VmProtectReq {
        addr: 0x4000_1000,
        len: 0x1000,
        prot: VmProt::READ | VmProt::USER,
    };
    vm_protect(&prot_req).expect("vm_protect failed");
    println!("Downgraded middle page to RO");

    // 4. Verify RO page is still readable
    let v = unsafe { ptr.add(0x1000).read_volatile() };
    if v != 0x43 {
        panic!("Read wrong value from RO page: 0x{:x}", v);
    }
    println!("Verified RO page readable");

    // 5. Restore RW
    let prot_req_rw = VmProtectReq {
        addr: 0x4000_1000,
        len: 0x1000,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
    };
    vm_protect(&prot_req_rw).expect("vm_protect failed");
    println!("Restored middle page to RW");

    // 6. Verify writable again
    unsafe {
        ptr.add(0x1000).write_volatile(0x45);
    }
    let v2 = unsafe { ptr.add(0x1000).read_volatile() };
    if v2 != 0x45 {
        panic!("Read wrong value after restoring RW: 0x{:x}", v2);
    }
    println!("Verified middle page writable again");

    println!("--- test_vm_protect finished successfully ---");
    stem::syscall::exit(0);
}
