//@ run-pass
//@ needs-unwind

#![allow(dropping_references, dropping_copy_types)]

// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]

static mut CHECK: usize = 0;

struct DropChecker(usize);

impl Drop for DropChecker {
    fn drop(&mut self) {
        unsafe {
            if CHECK != self.0 - 1 {
                panic!("Found {}, should have found {}", CHECK, self.0 - 1);
            }
            CHECK = self.0;
        }
    }
}

macro_rules! check_drops {
    ($l:literal) => {
        unsafe { assert_eq!(CHECK, $l) }
    };
}

struct DropPanic;

impl Drop for DropPanic {
    fn drop(&mut self) {
        panic!()
    }
}

fn value_zero() {
    unsafe { CHECK = 0 };
    let foo = DropChecker(1);
    let v: [DropChecker; 0] = [foo; 0];
    check_drops!(1);
    std::mem::drop(v);
    check_drops!(1);
}

fn value_one() {
    unsafe { CHECK = 0 };
    let foo = DropChecker(1);
    let v: [DropChecker; 1] = [foo; 1];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

const DROP_CHECKER: DropChecker = DropChecker(1);

fn const_zero() {
    unsafe { CHECK = 0 };
    let v: [DropChecker; 0] = [DROP_CHECKER; 0];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(0);
}

fn const_one() {
    unsafe { CHECK = 0 };
    let v: [DropChecker; 1] = [DROP_CHECKER; 1];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

fn const_generic_zero<const N: usize>() {
    unsafe { CHECK = 0 };
    let v: [DropChecker; N] = [DROP_CHECKER; N];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(0);
}

fn const_generic_one<const N: usize>() {
    unsafe { CHECK = 0 };
    let v: [DropChecker; N] = [DROP_CHECKER; N];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

// Make sure that things are allowed to promote as expected

fn allow_promote() {
    unsafe { CHECK = 0 };
    let foo = DropChecker(1);
    let v: &'static [DropChecker; 0] = &[foo; 0];
    check_drops!(1);
    std::mem::drop(v);
    check_drops!(1);
}

// Verify that unwinding in the drop causes the right things to drop in the right order
fn on_unwind() {
    unsafe { CHECK = 0 };
    std::panic::catch_unwind(|| {
        let panic = DropPanic;
        let _local = DropChecker(2);
        let _v = (DropChecker(1), [panic; 0]);
        std::process::abort();
    })
    .unwrap_err();
    check_drops!(2);
}

fn main() {
    value_zero();
    value_one();
    const_zero();
    const_one();
    const_generic_zero::<0>();
    const_generic_one::<1>();
    allow_promote();
    on_unwind();
}
