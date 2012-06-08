import libc::uintptr_t;

class frame {
    let fp: uintptr_t;

    new(fp: uintptr_t) {
        self.fp = fp;
    }
}

fn walk_stack(visit: fn(frame) -> bool) {
    frame_address { |frame_pointer|
        let frame_address = unsafe {
            unsafe::reinterpret_cast(frame_pointer)
        };
        visit(frame(frame_address));
    }
}

#[test]
fn test() {
    for walk_stack { |frame|
        #debug("frame: %x", frame.fp);
        // breakpoint();
    }
}

fn breakpoint() {
    rustrt::rust_dbg_breakpoint()
}

fn frame_address(f: fn(*u8)) {
    rusti::frame_address(f)
}

native mod rustrt {
    fn rust_dbg_breakpoint();
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn frame_address(f: fn(*u8));
}
