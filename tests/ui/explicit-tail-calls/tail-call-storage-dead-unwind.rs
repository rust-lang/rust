//@ run-pass
//@ ignore-backends: gcc
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

// Test that explicit tail calls work correctly when there are StorageDead
// and ForLint drops in the unwind path. This ensures that `unwind_to` is
// correctly adjusted in `break_for_tail_call` when encountering StorageDead
// and ForLint drops, matching the behavior in `build_scope_drops`.

#[allow(dead_code)]
struct Droppable(i32);

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn tail_call_with_storage_dead() {
    // These will have StorageDead drops (non-drop types)
    let _a = 42i32;
    let _b = true;
    let _c = 10u8;
    
    // This will have a Value drop (drop type)
    let _d = Droppable(1);
    
    // Tail call - if unwind_to isn't adjusted for StorageDead drops,
    // the debug assert will fail when processing the Value drop
    become next_function();
}

fn tail_call_with_mixed_drops() {
    // StorageDead drop
    let _storage = 100i32;
    
    // Value drop
    let _value = Droppable(2);
    
    // Another StorageDead drop
    let _storage2 = 200i32;
    
    // Another Value drop
    let _value2 = Droppable(3);
    
    // Tail call - tests that unwind_to is adjusted correctly
    // for both StorageDead and Value drops in sequence
    become next_function();
}

fn tail_call_with_storage_before_value() {
    // Multiple StorageDead drops before a Value drop
    let _s1 = 1i32;
    let _s2 = 2i32;
    let _s3 = 3i32;
    
    // Value drop - if StorageDead drops aren't handled,
    // unwind_to will point to wrong node and assert fails
    let _v = Droppable(4);
    
    become next_function();
}

fn next_function() {}

fn main() {
    tail_call_with_storage_dead();
    tail_call_with_mixed_drops();
    tail_call_with_storage_before_value();
}
