// This should fail even without validation or Stacked Borrows.
//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows -Cdebug-assertions=no

fn main() {
    // Make sure we notice when a u16 is loaded at offset 1 into a u8 allocation.
    // (This would be missed if u8 allocations are *always* at odd addresses.)
    //
    // Try many times as this might work by chance.
    for _ in 0..20 {
        let x = [0u8; 4];
        let ptr = x.as_ptr().wrapping_offset(1).cast::<u16>();
        let _val = unsafe { *ptr }; //~ERROR: but alignment
    }
}
