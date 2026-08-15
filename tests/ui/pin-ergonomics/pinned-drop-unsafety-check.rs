#![no_std]
#![no_core]
#![feature(no_core, pin_ergonomics, lang_items)]
#![allow(incomplete_features)]

// This checks that calling `Drop::pin_drop` and `Drop::drop` explicitly is unsafe
// and requires unsafe block.
// Note that this tiny-`core` library ignores `Unpin` related stuffs as we don't care about that,
// and thus the `Pin` type is just simply a wrapper around a pointer.

#[lang = "drop"]
trait Drop {
    fn drop(&mut self) {
        Self::pin_drop(Pin { pointer: self });
        //~^ ERROR call `Drop::pin_drop` explicitly is unsafe and requires unsafe block
        unsafe { Self::pin_drop(Pin { pointer: self }) }; // ok
    }

    fn pin_drop(self: Pin<&mut Self>) {
        Self::drop(self.pointer);
        //~^ ERROR call `Drop::drop` explicitly is unsafe and requires unsafe block
        unsafe { Self::drop(self.pointer) }; // ok
    }
}

#[lang = "pin"]
// This is a dummy `Pin` type that is just simply a wrapper around a pointer.
struct Pin<T> {
    pointer: T,
}

#[lang = "deref"]
trait Deref: PointeeSized {
    #[lang = "deref_target"]
    type Target: PointeeSized;

    fn deref(&self) -> &Self::Target;
}

#[lang = "deref_mut"]
trait DerefMut: Deref + PointeeSized {
    fn deref_mut(&mut self) -> &mut Self::Target;
}

impl<Ptr: Deref> Deref for Pin<Ptr> {
    type Target = Ptr::Target;

    fn deref(&self) -> &Self::Target {
        &*self.pointer
    }
}

// skip the `Unpin` check, as this test doesn't care about that.
impl<Ptr: DerefMut> DerefMut for Pin<Ptr> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.pointer
    }
}

impl<T: PointeeSized> Deref for &mut T {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self
    }
}


#[lang = "copy"]
trait Copy {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "legacy_receiver"]
trait LegacyReceiver: PointeeSized {}

impl<T: PointeeSized> LegacyReceiver for &T {}
impl<T: PointeeSized> LegacyReceiver for &mut T {}
impl<Ptr: LegacyReceiver> LegacyReceiver for Pin<Ptr> {}

fn main() {}
