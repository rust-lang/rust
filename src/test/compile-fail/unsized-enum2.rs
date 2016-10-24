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
    VA(W), //~ ERROR `W: std::marker::Sized` is not satisfied
    VB{x: X}, //~ ERROR `X: std::marker::Sized` is not satisfied
    VC(isize, Y), //~ ERROR `Y: std::marker::Sized` is not satisfied
    VD{u: isize, x: Z}, //~ ERROR `Z: std::marker::Sized` is not satisfied

    // slice / str
    VE([u8]), //~ ERROR `[u8]: std::marker::Sized` is not satisfied
    VF{x: str}, //~ ERROR `str: std::marker::Sized` is not satisfied
    VG(isize, [f32]), //~ ERROR `[f32]: std::marker::Sized` is not satisfied
    VH{u: isize, x: [u32]}, //~ ERROR `[u32]: std::marker::Sized` is not satisfied

    // unsized struct
    VI(Path1), //~ ERROR `PathHelper1 + 'static: std::marker::Sized` is not satisfied
    VJ{x: Path2}, //~ ERROR `PathHelper2 + 'static: std::marker::Sized` is not satisfied
    VK(isize, Path3), //~ ERROR `PathHelper3 + 'static: std::marker::Sized` is not satisfied
    VL{u: isize, x: Path4}, //~ ERROR `PathHelper4 + 'static: std::marker::Sized` is not satisfied

    // plain trait
    VM(Foo),  //~ ERROR `Foo + 'static: std::marker::Sized` is not satisfied
    VN{x: Bar}, //~ ERROR `Bar + 'static: std::marker::Sized` is not satisfied
    VO(isize, FooBar), //~ ERROR `FooBar + 'static: std::marker::Sized` is not satisfied
    VP{u: isize, x: BarFoo}, //~ ERROR `BarFoo + 'static: std::marker::Sized` is not satisfied

    // projected
    VQ(<&'static [i8] as Deref>::Target), //~ ERROR `[i8]: std::marker::Sized` is not satisfied
    VR{x: <&'static [char] as Deref>::Target},
    //~^ ERROR `[char]: std::marker::Sized` is not satisfied
    VS(isize, <&'static [f64] as Deref>::Target),
    //~^ ERROR `[f64]: std::marker::Sized` is not satisfied
    VT{u: isize, x: <&'static [i32] as Deref>::Target},
    //~^ ERROR `[i32]: std::marker::Sized` is not satisfied
}


fn main() { }

