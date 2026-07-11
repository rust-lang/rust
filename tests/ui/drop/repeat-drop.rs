//@ run-pass
//@ needs-unwind

#![allow(dropping_references, dropping_copy_types)]

static mut CHECK: usize = 0;

fn check() -> usize {
    unsafe { (&raw const CHECK).read() }
}

fn set_check(value: usize) {
    unsafe {
        (&raw mut CHECK).write(value);
    }
}

struct DropChecker(usize);

impl Drop for DropChecker {
    fn drop(&mut self) {
        let old = check();
        if old != self.0 - 1 {
            panic!("Found {}, should have found {}", old, self.0 - 1);
        }
        set_check(self.0);
    }
}

macro_rules! check_drops {
    ($l:literal) => {
        assert_eq!(check(), $l)
    };
}

struct DropPanic;

impl Drop for DropPanic {
    fn drop(&mut self) {
        panic!()
    }
}

fn value_zero() {
    set_check(0);
    let foo = DropChecker(1);
    let v: [DropChecker; 0] = [foo; 0];
    check_drops!(1);
    std::mem::drop(v);
    check_drops!(1);
}

fn value_one() {
    set_check(0);
    let foo = DropChecker(1);
    let v: [DropChecker; 1] = [foo; 1];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

const DROP_CHECKER: DropChecker = DropChecker(1);

fn const_zero() {
    set_check(0);
    let v: [DropChecker; 0] = [DROP_CHECKER; 0];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(0);
}

fn const_one() {
    set_check(0);
    let v: [DropChecker; 1] = [DROP_CHECKER; 1];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

fn const_generic_zero<const N: usize>() {
    set_check(0);
    let v: [DropChecker; N] = [DROP_CHECKER; N];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(0);
}

fn const_generic_one<const N: usize>() {
    set_check(0);
    let v: [DropChecker; N] = [DROP_CHECKER; N];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

// Make sure that things are allowed to promote as expected

fn allow_promote() {
    set_check(0);
    let foo = DropChecker(1);
    let v: &'static [DropChecker; 0] = &[foo; 0];
    check_drops!(1);
    std::mem::drop(v);
    check_drops!(1);
}

// Verify that unwinding in the drop causes the right things to drop in the right order
fn on_unwind() {
    set_check(0);
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
