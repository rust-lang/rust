//@ run-rustfix

// The `refining_impl_trait` suggestion must wrap a `+`-joined `impl Trait` in
// parens when it sits after a prefix type constructor (`&`, `&mut`, `*const`,
// `*mut`), so the suggestion is parser-correct. Covers multi-trait, `?Sized`,
// lifetime, projection-inlined and projection-only opaques, nested `&&`, and
// raw-pointer prefixes. Bare-position `impl A + B` must stay unwrapped.
//
// https://github.com/rust-lang/rust/issues/144401

#![allow(unused)]
#![deny(refining_impl_trait)]

pub trait BarA {}
impl<T: ?Sized> BarA for T {}

pub trait BarB {}
impl<T> BarB for T {}

pub trait Iter {
    type Item;
}
impl<T: ?Sized> Iter for T {
    type Item = u8;
}

pub struct Fool;

// (1)
pub trait BehindRef {
    fn bar(&self) -> &(impl BarA + BarB);
}
impl BehindRef for Fool {
    fn bar(&self) -> &Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        &Fool
    }
}

// (2)
pub trait BehindRefUnsized {
    fn bar_unsized(&self) -> &(impl BarA + ?Sized);
}
impl BehindRefUnsized for Fool {
    fn bar_unsized(&self) -> &Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        &Fool
    }
}

// (3)
pub trait BehindRefMut {
    fn bar_mut(&mut self) -> &mut (impl BarA + BarB);
}
impl BehindRefMut for Fool {
    fn bar_mut(&mut self) -> &mut Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        self
    }
}

// (4)
pub trait BehindRefLifetime<'a> {
    fn bar_lt(&'a self) -> &'a (impl BarA + 'a);
}
impl<'a> BehindRefLifetime<'a> for Fool {
    fn bar_lt(&'a self) -> &'a Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        &Fool
    }
}

// (5)
pub trait BehindRefProjAndTrait {
    fn bar_proj_trait(&self) -> &(impl Iter<Item = u8> + BarB);
}
impl BehindRefProjAndTrait for Fool {
    fn bar_proj_trait(&self) -> &Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        &Fool
    }
}

// (6) - no-wrap control: projection inlined, single joinable.
pub trait BehindRefProjOnly {
    fn bar_proj_only(&self) -> &impl Iter<Item = u8>;
}
impl BehindRefProjOnly for Fool {
    fn bar_proj_only(&self) -> &Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        &Fool
    }
}

// (7) - nested ref: only the inner `&` wraps.
pub trait BehindRefNested {
    fn bar_nested(&self) -> &&(impl BarA + BarB);
}
impl BehindRefNested for Fool {
    fn bar_nested(&self) -> &&Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        &&Fool
    }
}

// (8) - raw-pointer prefix generalization (`*const`).
pub trait BehindRawPtr {
    fn baz_raw() -> *const (impl BarA + BarB);
}
impl BehindRawPtr for Fool {
    fn baz_raw() -> *const Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        core::ptr::null::<Fool>()
    }
}

// (8b) - same generalization, `*mut` variant.
pub trait BehindRawPtrMut {
    fn baz_raw_mut() -> *mut (impl BarA + BarB);
}
impl BehindRawPtrMut for Fool {
    fn baz_raw_mut() -> *mut Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        core::ptr::null_mut::<Fool>()
    }
}

// Bare position must NOT pick up parens.
pub trait BarePosition {
    fn baz() -> impl BarA + BarB;
}
impl BarePosition for Fool {
    fn baz() -> Fool {
        //~^ ERROR impl trait in impl method signature does not match trait method signature
        Fool
    }
}

fn main() {}
