//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmir-move-elimination

// Check that valid calls preserve argument values without UB or leaked allocations.

#![feature(core_intrinsics, custom_mir, fn_traits, unboxed_closures)]
use std::intrinsics::mir::*;

// An earlier copy must complete before a later move frees the same source.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn copy_before_move() {
    mir! {
        let ptr: *const [u64; 4];
        {
            let value = [1u64; 4];
            // Force the source into memory to exercise snapshotting.
            ptr = &raw const value;
            Call(RET = compare(value, Move(value)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}
fn compare(a: [u64; 4], b: [u64; 4]) {
    assert_eq!(a, [1; 4]);
    assert_eq!(a, b);
}

// Rust-call untupling must preserve every field of the moved tuple.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn call_once<F: FnOnce([u64; 4], [u64; 4]) -> u64>(f: F, args: ([u64; 4], [u64; 4])) -> u64 {
    mir! {
        let f_ptr: *const F;
        let args_ptr: *const ([u64; 4], [u64; 4]);
        {
            // Force both moved arguments into memory.
            f_ptr = &raw const f;
            args_ptr = &raw const args;
            Call(RET = F::call_once(Move(f), Move(args)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

// Capturing a projected move must not retain a dependency on its pointer local.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn projected_before_pointer() {
    mir! {
        let ptr: *mut u32;
        {
            let value = 42u32;
            ptr = &raw mut value;
            Call(RET = value_and_pointer(Move(*ptr), Move(ptr)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}
fn value_and_pointer(value: u32, _: *mut u32) {
    assert_eq!(value, 42);
}

fn main() {
    projected_before_pointer();
    copy_before_move();

    let captured = Box::new([10u64; 4]);
    let closure = move |a: [u64; 4], b: [u64; 4]| {
        assert_eq!(a, [1; 4]);
        assert_eq!(b, [2; 4]);
        a.iter().sum::<u64>() + b.iter().sum::<u64>() + captured.iter().sum::<u64>()
    };
    assert_eq!(call_once(closure, ([1; 4], [2; 4])), 52);

    // A call may unwind without having allocated its return local.
    std::panic::set_hook(Box::new(|_| {}));
    assert!(std::panic::catch_unwind(|| panic!("caught")).is_err());
}
