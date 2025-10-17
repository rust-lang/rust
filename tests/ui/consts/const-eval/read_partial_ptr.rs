//! Ensure we error when trying to load from a pointer whose provenance has been messed with.
//@ ignore-test: disabled due to <https://github.com/rust-lang/rust/issues/146291>

const PARTIAL_OVERWRITE: () = {
    let mut p = &42;
    // Overwrite one byte with a no-provenance value.
    unsafe {
        let ptr: *mut _ = &mut p;
        *(ptr as *mut u8) = 123;
    }
    let x = *p; //~ ERROR: unable to read parts of a pointer
};

const PTR_BYTES_SWAP: () = {
    let mut p = &42;
    // Swap the first two bytes.
    unsafe {
        let ptr = &mut p as *mut _ as *mut std::mem::MaybeUninit<u8>;
        let byte0 = ptr.read();
        let byte1 = ptr.add(1).read();
        ptr.write(byte1);
        ptr.add(1).write(byte0);
    }
    let x = *p; //~ ERROR: unable to read parts of a pointer
};

const PTR_BYTES_REPEAT: () = {
    let mut p = &42;
    // Duplicate the first byte over the second.
    unsafe {
        let ptr = &mut p as *mut _ as *mut std::mem::MaybeUninit<u8>;
        let byte0 = ptr.read();
        ptr.add(1).write(byte0);
    }
    let x = *p; //~ ERROR: unable to read parts of a pointer
};

const PTR_BYTES_MIX: () = {
    let mut p = &42;
    let q = &43;
    // Overwrite the first byte of p with the first byte of q.
    unsafe {
        let ptr = &mut p as *mut _ as *mut std::mem::MaybeUninit<u8>;
        let qtr = &q as *const _ as *const std::mem::MaybeUninit<u8>;
        ptr.write(qtr.read());
    }
    let x = *p; //~ ERROR: unable to read parts of a pointer
};

fn main() {}
