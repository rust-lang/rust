#![allow(unused)]
use std::{cell::UnsafeCell, mem, sync::atomic::AtomicI32};

fn main() {
    unsafe {
        mem::transmute::<&i32, &UnsafeCell<i32>>(&42);
        //~^ WARNING transmuting from a type without interior mutability to a type with interior mutability
        //~| HELP `UnsafeCell<i32>` has interior mutability
        // It's an error to transmute to a type containing unsafe cell
        mem::transmute::<&i32, &AtomicI32>(&42);
        //~^ WARNING transmuting from a type without interior mutability to a type with interior mutability
        //~| HELP `AtomicI32` has interior mutability
        // mutable_transmutes triggers before

        // This one is here because & -> &mut is worse, to assert that this one triggers.
        mem::transmute::<&i32, &mut UnsafeCell<i32>>(&42);
        //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
    };
}
