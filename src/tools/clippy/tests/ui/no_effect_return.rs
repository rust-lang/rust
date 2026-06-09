//@no-rustfix: overlapping suggestions
#![allow(clippy::unused_unit, dead_code, unused)]
#![no_main]

use std::ops::ControlFlow;

fn a() -> u32 {
    {
        0u32;
        //~^ no_effect
    }
    0
}

async fn b() -> u32 {
    {
        0u32;
        //~^ no_effect
    }
    0
}

type C = i32;
async fn c() -> C {
    {
        0i32 as C;
        //~^ no_effect
    }
    0
}

fn d() -> u128 {
    {
        // not last stmt
        0u128;
        //~^ no_effect

        println!("lol");
    }
    0
}

fn e() -> u32 {
    {
        // mismatched types
        0u16;
        //~^ no_effect
    }
    0
}

fn f() -> [u16; 1] {
    {
        [1u16];
        //~^ no_effect
    }
    [1]
}

fn g() -> ControlFlow<()> {
    {
        ControlFlow::Break::<()>(());
        //~^ no_effect
    }
    ControlFlow::Continue(())
}

fn h() -> Vec<u16> {
    {
        // function call, so this won't trigger `no_effect`. not an issue with this change, but the
        // lint itself (but also not really.)
        vec![0u16];
    }
    vec![]
}

fn i() -> () {
    {
        // does not suggest on function with explicit unit return type
        ();
        //~^ no_effect
    }
    ()
}

fn j() {
    {
        // does not suggest on function without explicit return type
        ();
        //~^ no_effect
    }
    ()
}
