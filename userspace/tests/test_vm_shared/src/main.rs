#![no_std]
#![no_main]
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt, VmUnmapReq};
use stem::println;
use stem::syscall::{memfd_create, vfs_close};
use stem::vm::{vm_map, vm_unmap};

const SIZE: usize = 4096;

#[stem::main]
fn main(_arg0: usize) -> ! {
    println!("--- test_vm_shared starting ---");

    let fd = memfd_create("test-vm-shared", SIZE).expect("memfd_create failed");

    let map_shared_rw_a = vm_map(&VmMapReq {
        addr_hint: 0,
        len: SIZE,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::SHARED,
        backing: VmBacking::File { fd, offset: 0 },
    })
    .expect("shared map a failed");

    let map_shared_rw_b = vm_map(&VmMapReq {
        addr_hint: 0,
        len: SIZE,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::SHARED,
        backing: VmBacking::File { fd, offset: 0 },
    })
    .expect("shared map b failed");

    unsafe {
        let a = map_shared_rw_a.addr as *mut u8;
        let b = map_shared_rw_b.addr as *const u8;
        a.write_volatile(0x5A);
        let seen = b.read_volatile();
        assert_eq!(seen, 0x5A, "shared map did not reflect peer write");
    }
    println!("shared mapping coherence: PASS");

    let map_private = vm_map(&VmMapReq {
        addr_hint: 0,
        len: SIZE,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::PRIVATE,
        backing: VmBacking::File { fd, offset: 0 },
    })
    .expect("private map failed");

    unsafe {
        let p = map_private.addr as *mut u8;
        let s = map_shared_rw_a.addr as *const u8;
        p.write_volatile(0xA5);
        let seen_shared = s.read_volatile();
        assert_eq!(
            seen_shared, 0x5A,
            "private map write leaked into shared mapping"
        );
    }
    println!("private mapping isolation: PASS");

    vm_unmap(&VmUnmapReq {
        addr: map_shared_rw_a.addr,
        len: SIZE,
    })
    .expect("unmap shared a failed");
    vm_unmap(&VmUnmapReq {
        addr: map_shared_rw_b.addr,
        len: SIZE,
    })
    .expect("unmap shared b failed");
    vm_unmap(&VmUnmapReq {
        addr: map_private.addr,
        len: SIZE,
    })
    .expect("unmap private failed");

    vfs_close(fd).expect("close memfd failed");
    println!("--- test_vm_shared: PASS ---");
    stem::syscall::exit(0);
}
