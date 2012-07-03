// NB: Don't rely on other core mods here as this has to move into the rt

import unsafe::reinterpret_cast;
import ptr::offset;
import sys::size_of;

type word = uint;

class frame {
    let fp: *word;

    new(fp: *word) {
        self.fp = fp;
    }
}

fn walk_stack(visit: fn(frame) -> bool) {

    #debug("beginning stack walk");

    do frame_address |frame_pointer| {
        let mut frame_address: *word = unsafe {
            reinterpret_cast(frame_pointer)
        };
        loop {
            let fr = frame(frame_address);

            #debug("frame: %x", unsafe { reinterpret_cast(fr.fp) });
            visit(fr);

            unsafe {
                let next_fp: **word = reinterpret_cast(frame_address);
                frame_address = *next_fp;
                if *frame_address == 0u {
                    #debug("encountered task_start_wrapper. ending walk");
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
        if i == 0 { ret }

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
    fn rust_dbg_breakpoint();
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn frame_address(f: fn(*u8));
}
