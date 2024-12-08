// Because drop recursion can only be detected after drop elaboration which
// happens for codegen:
//@ build-fail

#![deny(unconditional_recursion)]
#![allow(dead_code)]

pub struct RecursiveDrop;

impl Drop for RecursiveDrop {
    fn drop(&mut self) { //~ ERROR function cannot return without recursing
        let _ = RecursiveDrop;
    }
}

#[derive(Default)]
struct NotRecursiveDrop1;

impl Drop for NotRecursiveDrop1 {
    fn drop(&mut self) {
        // Before drop elaboration, the MIR can look like a recursive drop will
        // occur. But it will not, since forget() prevents drop() from running.
        let taken = std::mem::take(self);
        std::mem::forget(taken);
    }
}

struct NotRecursiveDrop2;

impl Drop for NotRecursiveDrop2 {
    fn drop(&mut self) {
        // Before drop elaboration, the MIR can look like a recursive drop will
        // occur. But it will not, since this will panic.
        std::panic::panic_any(NotRecursiveDrop2);
    }
}

fn main() {}
