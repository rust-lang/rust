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

struct Path1(dyn PathHelper1);
struct Path2(dyn PathHelper2);
struct Path3(dyn PathHelper3);
struct Path4(dyn PathHelper4);

enum E<W: ?Sized, X: ?Sized, Y: ?Sized, Z: ?Sized> {
    // parameter
    VA(W),
    //~^ ERROR the size for values of type
    VB{x: X},
    //~^ ERROR the size for values of type
    VC(isize, Y),
    //~^ ERROR the size for values of type
    VD{u: isize, x: Z},
    //~^ ERROR the size for values of type

    // slice / str
    VE([u8]),
    //~^ ERROR the size for values of type
    VF{x: str},
    //~^ ERROR the size for values of type
    VG(isize, [f32]),
    //~^ ERROR the size for values of type
    VH{u: isize, x: [u32]},
    //~^ ERROR the size for values of type

    // unsized struct
    VI(Path1),
    //~^ ERROR the size for values of type
    VJ{x: Path2},
    //~^ ERROR the size for values of type
    VK(isize, Path3),
    //~^ ERROR the size for values of type
    VL{u: isize, x: Path4},
    //~^ ERROR the size for values of type

    // plain trait
    VM(dyn Foo),
    //~^ ERROR the size for values of type
    VN{x: dyn Bar},
    //~^ ERROR the size for values of type
    VO(isize, dyn FooBar),
    //~^ ERROR the size for values of type
    VP{u: isize, x: dyn BarFoo},
    //~^ ERROR the size for values of type

    // projected
    VQ(<&'static [i8] as Deref>::Target),
    //~^ ERROR the size for values of type
    VR{x: <&'static [char] as Deref>::Target},
    //~^ ERROR the size for values of type
    VS(isize, <&'static [f64] as Deref>::Target),
    //~^ ERROR the size for values of type
    VT{u: isize, x: <&'static [i32] as Deref>::Target},
    //~^ ERROR the size for values of type
}


fn main() { }
