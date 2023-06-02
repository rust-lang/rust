#![allow(clippy::let_unit_value, clippy::unnecessary_cast)]

fn main() {
    let x: [i32; 3] = [1_i32, 2, 3];
    let r_x = &x;
    // Check casting through multiple bindings
    // Because it's separate, it does not check the cast back to something of the same size
    let a = r_x as *const [i32];
    let b = a as *const [u8];
    let c = b as *const [u32];

    // loses data
    let loss = r_x as *const [i32] as *const [u8];

    // Cast back to same size but different type loses no data, just type conversion
    // This is weird code but there's no reason for this lint specifically to fire *twice* on it
    let restore = r_x as *const [i32] as *const [u8] as *const [u32];

    // Check casting through blocks is detected
    let loss_block_1 = { r_x as *const [i32] } as *const [u8];
    let loss_block_2 = {
        let _ = ();
        r_x as *const [i32]
    } as *const [u8];

    // Check that resources of the same size are detected through blocks
    let restore_block_1 = { r_x as *const [i32] } as *const [u8] as *const [u32];
    let restore_block_2 = { ({ r_x as *const [i32] }) as *const [u8] } as *const [u32];
    let restore_block_3 = {
        let _ = ();
        ({
            let _ = ();
            r_x as *const [i32]
        }) as *const [u8]
    } as *const [u32];

    // Check that the result of a long chain of casts is detected
    let long_chain_loss = r_x as *const [i32] as *const [u32] as *const [u16] as *const [i8] as *const [u8];
    let long_chain_restore =
        r_x as *const [i32] as *const [u32] as *const [u16] as *const [i8] as *const [u8] as *const [u32];
}

// foo and foo2 should not fire, they're the same size
fn foo(x: *mut [u8]) -> *mut [u8] {
    x as *mut [u8]
}

fn foo2(x: *mut [u8]) -> *mut [u8] {
    x as *mut _
}

// Test that casts as part of function returns work
fn bar(x: *mut [u16]) -> *mut [u8] {
    x as *mut [u8]
}

fn uwu(x: *mut [u16]) -> *mut [u8] {
    x as *mut _
}

fn bar2(x: *mut [u16]) -> *mut [u8] {
    x as _
}

// constify
fn bar3(x: *mut [u16]) -> *const [u8] {
    x as _
}

// unconstify
fn bar4(x: *const [u16]) -> *mut [u8] {
    x as _
}

// function returns plus blocks
fn blocks(x: *mut [u16]) -> *mut [u8] {
    ({ x }) as _
}

fn more_blocks(x: *mut [u16]) -> *mut [u8] {
    { ({ x }) as _ }
}
