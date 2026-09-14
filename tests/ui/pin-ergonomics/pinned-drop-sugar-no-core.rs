//@ check-pass

#![no_std]
#![no_core]
#![no_main]
#![feature(no_core, pin_ergonomics, lang_items)]
#![allow(incomplete_features)]

#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "copy"]
trait Copy {}

#[lang = "legacy_receiver"]
trait LegacyReceiver: PointeeSized {}

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

#[lang = "pin"]
struct Pin<T> {
    pointer: T,
}

impl<T: PointeeSized> LegacyReceiver for &T {}
impl<T: PointeeSized> LegacyReceiver for &mut T {}
impl<Ptr: LegacyReceiver> LegacyReceiver for Pin<Ptr> {}

impl<Ptr: Deref> Deref for Pin<Ptr> {
    type Target = Ptr::Target;

    fn deref(&self) -> &Self::Target {
        &*self.pointer
    }
}

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

struct LocalDrop;

impl Drop for LocalDrop {
    fn drop(&pin mut self) {}
}

#[lang = "drop"]
trait Drop {
    fn drop(&mut self) {}
    fn pin_drop(self: Pin<&mut Self>) {}
}
