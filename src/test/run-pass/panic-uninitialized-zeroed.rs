// ignore-wasm32-bare always compiled as panic=abort right now and this requires unwinding
// This test checks that instantiating an uninhabited type via `mem::{uninitialized,zeroed}` results
// in a runtime panic.

#![feature(never_type)]

use std::{mem, panic};

#[allow(dead_code)]
struct Foo {
    x: u8,
    y: !,
}

enum Bar {}

fn main() {
    unsafe {
        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<!>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type ! using mem::uninitialized"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<!>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type ! using mem::zeroed"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<Foo>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo using mem::uninitialized"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<Foo>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo using mem::zeroed"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<Bar>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar using mem::uninitialized"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<Bar>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar using mem::zeroed"
            })),
            Some(true)
        );
    }
}
