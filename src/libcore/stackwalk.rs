#[doc(hidden)]; // FIXME #3538

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cast::reinterpret_cast;
use ptr::offset;
use sys::size_of;

type Word = uint;

struct Frame {
    fp: *Word
}

fn Frame(fp: *Word) -> Frame {
    Frame {
        fp: fp
    }
}

fn walk_stack(visit: fn(Frame) -> bool) {

    debug!("beginning stack walk");

    do frame_address |frame_pointer| {
        let mut frame_address: *Word = unsafe {
            reinterpret_cast(&frame_pointer)
        };
        loop {
            let fr = Frame(frame_address);

            debug!("frame: %x", unsafe { reinterpret_cast(&fr.fp) });
            visit(fr);

            unsafe {
                let next_fp: **Word = reinterpret_cast(&frame_address);
                frame_address = *next_fp;
                if *frame_address == 0u {
                    debug!("encountered task_start_wrapper. ending walk");
                    // This is the task_start_wrapper_frame. There is
                    // no stack beneath it and it is a foreign frame.
                    break;
                }
            }
        }
    }
}

#[test]
fn test_simple() {
    for walk_stack |_frame| {
    }
}

#[test]
fn test_simple_deep() {
    fn run(i: int) {
        if i == 0 { return }

        for walk_stack |_frame| {
            unsafe {
                breakpoint();
            }
        }
        run(i - 1);
    }

    run(10);
}

fn breakpoint() {
    rustrt::rust_dbg_breakpoint()
}

fn frame_address(f: fn(*u8)) {
    rusti::frame_address(f)
}

extern mod rustrt {
    #[legacy_exports];
    fn rust_dbg_breakpoint();
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn frame_address(f: fn(*u8));
}
