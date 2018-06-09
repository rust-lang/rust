// Copyright 206 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

// Due to aggressive error message deduplication, we require 20 *different*
// unsized types (even Path and [u8] are considered the "same").

trait Foo {}
trait Bar {}
trait FooBar {}
trait BarFoo {}

trait PathHelper1 {}
trait PathHelper2 {}
trait PathHelper3 {}
trait PathHelper4 {}

struct Path1(PathHelper1);
struct Path2(PathHelper2);
struct Path3(PathHelper3);
struct Path4(PathHelper4);

enum E<W: ?Sized, X: ?Sized, Y: ?Sized, Z: ?Sized> {
    // parameter
    VA(W),
    //~^ ERROR `W` does not have a constant size known at compile-time
    VB{x: X},
    //~^ ERROR `X` does not have a constant size known at compile-time
    VC(isize, Y),
    //~^ ERROR `Y` does not have a constant size known at compile-time
    VD{u: isize, x: Z},
    //~^ ERROR `Z` does not have a constant size known at compile-time

    // slice / str
    VE([u8]),
    //~^ ERROR `[u8]` does not have a constant size known at compile-time
    VF{x: str},
    //~^ ERROR `str` does not have a constant size known at compile-time
    VG(isize, [f32]),
    //~^ ERROR `[f32]` does not have a constant size known at compile-time
    VH{u: isize, x: [u32]},
    //~^ ERROR `[u32]` does not have a constant size known at compile-time

    // unsized struct
    VI(Path1),
    //~^ ERROR `PathHelper1 + 'static` does not have a constant size known at compile-time
    VJ{x: Path2},
    //~^ ERROR `PathHelper2 + 'static` does not have a constant size known at compile-time
    VK(isize, Path3),
    //~^ ERROR `PathHelper3 + 'static` does not have a constant size known at compile-time
    VL{u: isize, x: Path4},
    //~^ ERROR `PathHelper4 + 'static` does not have a constant size known at compile-time

    // plain trait
    VM(Foo),
    //~^ ERROR `Foo + 'static` does not have a constant size known at compile-time
    VN{x: Bar},
    //~^ ERROR `Bar + 'static` does not have a constant size known at compile-time
    VO(isize, FooBar),
    //~^ ERROR `FooBar + 'static` does not have a constant size known at compile-time
    VP{u: isize, x: BarFoo},
    //~^ ERROR `BarFoo + 'static` does not have a constant size known at compile-time

    // projected
    VQ(<&'static [i8] as Deref>::Target),
    //~^ ERROR `[i8]` does not have a constant size known at compile-time
    VR{x: <&'static [char] as Deref>::Target},
    //~^ ERROR `[char]` does not have a constant size known at compile-time
    VS(isize, <&'static [f64] as Deref>::Target),
    //~^ ERROR `[f64]` does not have a constant size known at compile-time
    VT{u: isize, x: <&'static [i32] as Deref>::Target},
    //~^ ERROR `[i32]` does not have a constant size known at compile-time
}


fn main() { }

