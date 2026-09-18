//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmir-move-elimination

// Check that valid calls preserve argument values without UB or leaked allocations.

#![feature(core_intrinsics, custom_mir, fn_traits, unboxed_closures)]
use std::intrinsics::mir::*;

trait Consume {
    fn consume(self: Box<Self>) -> u64;
}
impl Consume for [u64; 4] {
    fn consume(self: Box<Self>) -> u64 {
        self.iter().sum()
    }
}

// Emulated intrinsics must consume whole-move arguments and clean up their storage.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn intrinsic(value: [u64; 4]) -> [u64; 4] {
    mir! {
        {
            Call(RET = std::intrinsics::black_box(Move(value)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

// An earlier copy must complete before a later move frees the same source.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn copy_before_move() {
    mir! {
        let ptr: *const [u64; 4];
        let unit: ();
        {
            let value = [1u64; 4];
            ptr = &raw const value;
            Call(unit = compare(value, Move(value)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}
fn compare(a: [u64; 4], b: [u64; 4]) {
    assert_eq!(a, b);
}

// Rust-call untupling must copy every field before freeing the tuple allocation,
// while also consuming the moved closure.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn call_once<F: FnOnce([u64; 4], [u64; 4]) -> u64>(f: F, args: ([u64; 4], [u64; 4])) -> u64 {
    mir! {
        let f_ptr: *const F;
        let args_ptr: *const ([u64; 4], [u64; 4]);
        {
            f_ptr = &raw const f;
            args_ptr = &raw const args;
            Call(RET = F::call_once(Move(f), Move(args)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

// Virtual receiver adjustment must preserve cleanup of the original moved local.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn virtual_call(receiver: Box<dyn Consume>) -> u64 {
    mir! {
        let ptr: *const Box<dyn Consume>;
        {
            // Force the actual call operand into memory, without an intermediate move.
            ptr = &raw const receiver;
            Call(RET = <dyn Consume as Consume>::consume(Move(receiver)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

// A call move must leave its local readable until the destination is evaluated.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn destination_uses_moved_pointer() -> u32 {
    mir! {
        let value: u32;
        let ptr: *mut u32;
        {
            value = 0;
            ptr = &raw mut value;
            Call(*ptr = pointer_arg(Move(ptr)), ReturnTo(done), UnwindContinue())
        }
        done = {
            RET = value;
            Return()
        }
    }
}
fn pointer_arg(_: *mut u32) -> u32 {
    42
}

// Later arguments can still dereference a pointer moved by an earlier argument.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn argument_uses_moved_pointer() {
    mir! {
        let ptr: *const u32;
        let unit: ();
        {
            let value = 42u32;
            ptr = &raw const value;
            Call(unit = pointer_and_value(Move(ptr), *ptr), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}
fn pointer_and_value(_: *const u32, value: u32) {
    assert_eq!(value, 42);
}

fn main() {
    argument_uses_moved_pointer();
    assert_eq!(destination_uses_moved_pointer(), 42);
    copy_before_move();

    let captured = Box::new([10u64; 4]);
    let closure = move |a: [u64; 4], b: [u64; 4]| {
        a.iter().sum::<u64>() + b.iter().sum::<u64>() + captured.iter().sum::<u64>()
    };
    assert_eq!(call_once(closure, ([1; 4], [2; 4])), 52);

    assert_eq!(virtual_call(Box::new([3; 4])), 12);
    assert_eq!(intrinsic([4; 4]), [4; 4]);

    // A call may unwind without having allocated its return local.
    std::panic::set_hook(Box::new(|_| {}));
    assert_eq!(std::panic::catch_unwind(|| 42).unwrap(), 42);
    assert!(std::panic::catch_unwind(|| panic!("caught")).is_err());
}
