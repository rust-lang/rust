// ignore-wasm32-bare always compiled as panic=abort right now and this requires unwinding
// This test checks that instantiating an uninhabited type via `mem::{uninitialized,zeroed}` results
// in a runtime panic.

#![feature(never_type)]
#![allow(deprecated)]

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
                s == "Attempted to instantiate uninhabited type !"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<!>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type !"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::MaybeUninit::<!>::uninit().assume_init()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type !"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<Foo>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<Foo>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::MaybeUninit::<Foo>::uninit().assume_init()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Foo"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::uninitialized::<Bar>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::zeroed::<Bar>()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar"
            })),
            Some(true)
        );

        assert_eq!(
            panic::catch_unwind(|| {
                mem::MaybeUninit::<Bar>::uninit().assume_init()
            }).err().and_then(|a| a.downcast_ref::<String>().map(|s| {
                s == "Attempted to instantiate uninhabited type Bar"
            })),
            Some(true)
        );
    }
}
