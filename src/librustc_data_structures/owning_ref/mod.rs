#![warn(missing_docs)]

/*!
# An owning reference.

This crate provides the _owning reference_ types `OwningRef` and `OwningRefMut`
that enables it to bundle a reference together with the owner of the data it points to.
This allows moving and dropping of a `OwningRef` without needing to recreate the reference.

This can sometimes be useful because Rust borrowing rules normally prevent
moving a type that has been moved from. For example, this kind of code gets rejected:

```rust,ignore
fn return_owned_and_referenced<'a>() -> (Vec<u8>, &'a [u8]) {
    let v = vec![1, 2, 3, 4];
    let s = &v[1..3];
    (v, s)
}
```

Even though, from a memory-layout point of view, this can be entirely safe
if the new location of the vector still lives longer than the lifetime `'a`
of the reference because the backing allocation of the vector does not change.

This library enables this safe usage by keeping the owner and the reference
bundled together in a wrapper type that ensure that lifetime constraint:

```rust
# extern crate owning_ref;
# use owning_ref::OwningRef;
# fn main() {
fn return_owned_and_referenced() -> OwningRef<Vec<u8>, [u8]> {
    let v = vec![1, 2, 3, 4];
    let or = OwningRef::new(v);
    let or = or.map(|v| &v[1..3]);
    or
}
# }
```

It works by requiring owner types to dereference to stable memory locations
and preventing mutable access to root containers, which in practice requires heap allocation
as provided by `Box<T>`, `Rc<T>`, etc.

Also provided are typedefs for common owner type combinations,
which allow for less verbose type signatures. For example, `BoxRef<T>` instead of `OwningRef<Box<T>, T>`.

The crate also provides the more advanced `OwningHandle` type,
which allows more freedom in bundling a dependent handle object
along with the data it depends on, at the cost of some unsafe needed in the API.
See the documentation around `OwningHandle` for more details.

# Examples

## Basics

```
extern crate owning_ref;
use owning_ref::BoxRef;

fn main() {
    // Create an array owned by a Box.
    let arr = Box::new([1, 2, 3, 4]) as Box<[i32]>;

    // Transfer into a BoxRef.
    let arr: BoxRef<[i32]> = BoxRef::new(arr);
    assert_eq!(&*arr, &[1, 2, 3, 4]);

    // We can slice the array without losing ownership or changing type.
    let arr: BoxRef<[i32]> = arr.map(|arr| &arr[1..3]);
    assert_eq!(&*arr, &[2, 3]);

    // Also works for Arc, Rc, String and Vec!
}
```

## Caching a reference to a struct field

```
extern crate owning_ref;
use owning_ref::BoxRef;

fn main() {
    struct Foo {
        tag: u32,
        x: u16,
        y: u16,
        z: u16,
    }
    let foo = Foo { tag: 1, x: 100, y: 200, z: 300 };

    let or = BoxRef::new(Box::new(foo)).map(|foo| {
        match foo.tag {
            0 => &foo.x,
            1 => &foo.y,
            2 => &foo.z,
            _ => panic!(),
        }
    });

    assert_eq!(*or, 200);
}
```

## Caching a reference to an entry in a vector

```
extern crate owning_ref;
use owning_ref::VecRef;

fn main() {
    let v = VecRef::new(vec![1, 2, 3, 4, 5]).map(|v| &v[3]);
    assert_eq!(*v, 4);
}
```

## Caching a subslice of a String

```
extern crate owning_ref;
use owning_ref::StringRef;

fn main() {
    let s = StringRef::new("hello world".to_owned())
        .map(|s| s.split(' ').nth(1).unwrap());

    assert_eq!(&*s, "world");
}
```

## Reference counted slices that share ownership of the backing storage

```
extern crate owning_ref;
use owning_ref::RcRef;
use std::rc::Rc;

fn main() {
    let rc: RcRef<[i32]> = RcRef::new(Rc::new([1, 2, 3, 4]) as Rc<[i32]>);
    assert_eq!(&*rc, &[1, 2, 3, 4]);

    let rc_a: RcRef<[i32]> = rc.clone().map(|s| &s[0..2]);
    let rc_b = rc.clone().map(|s| &s[1..3]);
    let rc_c = rc.clone().map(|s| &s[2..4]);
    assert_eq!(&*rc_a, &[1, 2]);
    assert_eq!(&*rc_b, &[2, 3]);
    assert_eq!(&*rc_c, &[3, 4]);

    let rc_c_a = rc_c.clone().map(|s| &s[1]);
    assert_eq!(&*rc_c_a, &4);
}
```

## Atomic reference counted slices that share ownership of the backing storage

```
extern crate owning_ref;
use owning_ref::ArcRef;
use std::sync::Arc;

fn main() {
    use std::thread;

    fn par_sum(rc: ArcRef<[i32]>) -> i32 {
        if rc.len() == 0 {
            return 0;
        } else if rc.len() == 1 {
            return rc[0];
        }
        let mid = rc.len() / 2;
        let left = rc.clone().map(|s| &s[..mid]);
        let right = rc.map(|s| &s[mid..]);

        let left = thread::spawn(move || par_sum(left));
        let right = thread::spawn(move || par_sum(right));

        left.join().unwrap() + right.join().unwrap()
    }

    let rc: Arc<[i32]> = Arc::new([1, 2, 3, 4]);
    let rc: ArcRef<[i32]> = rc.into();

    assert_eq!(par_sum(rc), 10);
}
```

## References into RAII locks

```
extern crate owning_ref;
use owning_ref::RefRef;
use std::cell::{RefCell, Ref};

fn main() {
    let refcell = RefCell::new((1, 2, 3, 4));
    // Also works with Mutex and RwLock

    let refref = {
        let refref = RefRef::new(refcell.borrow()).map(|x| &x.3);
        assert_eq!(*refref, 4);

        // We move the RAII lock and the reference to one of
        // the subfields in the data it guards here:
        refref
    };

    assert_eq!(*refref, 4);

    drop(refref);

    assert_eq!(*refcell.borrow(), (1, 2, 3, 4));
}
```

## Mutable reference

When the owned container implements `DerefMut`, it is also possible to make
a _mutable owning reference_. (E.g. with `Box`, `RefMut`, `MutexGuard`)

```
extern crate owning_ref;
use owning_ref::RefMutRefMut;
use std::cell::{RefCell, RefMut};

fn main() {
    let refcell = RefCell::new((1, 2, 3, 4));

    let mut refmut_refmut = {
        let mut refmut_refmut = RefMutRefMut::new(refcell.borrow_mut()).map_mut(|x| &mut x.3);
        assert_eq!(*refmut_refmut, 4);
        *refmut_refmut *= 2;

        refmut_refmut
    };

    assert_eq!(*refmut_refmut, 8);
    *refmut_refmut *= 2;

    drop(refmut_refmut);

    assert_eq!(*refcell.borrow(), (1, 2, 3, 16));
}
```
*/

use std::mem;
pub use stable_deref_trait::{StableDeref as StableAddress, CloneStableDeref as CloneStableAddress};

/// An owning reference.
///
/// This wraps an owner `O` and a reference `&T` pointing
/// at something reachable from `O::Target` while keeping
/// the ability to move `self` around.
///
/// The owner is usually a pointer that points at some base type.
///
/// For more details and examples, see the module and method docs.
pub struct OwningRef<O, T: ?Sized> {
    owner: O,
    reference: *const T,
}

/// An mutable owning reference.
///
/// This wraps an owner `O` and a reference `&mut T` pointing
/// at something reachable from `O::Target` while keeping
/// the ability to move `self` around.
///
/// The owner is usually a pointer that points at some base type.
///
/// For more details and examples, see the module and method docs.
pub struct OwningRefMut<O, T: ?Sized> {
    owner: O,
    reference: *mut T,
}

/// Helper trait for an erased concrete type an owner dereferences to.
/// This is used in form of a trait object for keeping
/// something around to (virtually) call the destructor.
pub trait Erased {}
impl<T> Erased for T {}

/// Helper trait for erasing the concrete type of what an owner dereferences to,
/// for example `Box<T> -> Box<Erased>`. This would be unneeded with
/// higher kinded types support in the language.
pub unsafe trait IntoErased<'a> {
    /// Owner with the dereference type substituted to `Erased`.
    type Erased;
    /// Perform the type erasure.
    fn into_erased(self) -> Self::Erased;
}

/// Helper trait for erasing the concrete type of what an owner dereferences to,
/// for example `Box<T> -> Box<Erased + Send>`. This would be unneeded with
/// higher kinded types support in the language.
pub unsafe trait IntoErasedSend<'a> {
    /// Owner with the dereference type substituted to `Erased + Send`.
    type Erased: Send;
    /// Perform the type erasure.
    fn into_erased_send(self) -> Self::Erased;
}

/// Helper trait for erasing the concrete type of what an owner dereferences to,
/// for example `Box<T> -> Box<Erased + Send + Sync>`. This would be unneeded with
/// higher kinded types support in the language.
pub unsafe trait IntoErasedSendSync<'a> {
    /// Owner with the dereference type substituted to `Erased + Send + Sync`.
    type Erased: Send + Sync;
    /// Perform the type erasure.
    fn into_erased_send_sync(self) -> Self::Erased;
}

/////////////////////////////////////////////////////////////////////////////
// OwningRef
/////////////////////////////////////////////////////////////////////////////

impl<O, T: ?Sized> OwningRef<O, T> {
    /// Creates a new owning reference from a owner
    /// initialized to the direct dereference of it.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRef;
    ///
    /// fn main() {
    ///     let owning_ref = OwningRef::new(Box::new(42));
    ///     assert_eq!(*owning_ref, 42);
    /// }
    /// ```
    pub fn new(o: O) -> Self
        where O: StableAddress,
              O: Deref<Target = T>,
    {
        OwningRef {
            reference: &*o,
            owner: o,
        }
    }

    /// Like `new`, but doesn’t require `O` to implement the `StableAddress` trait.
    /// Instead, the caller is responsible to make the same promises as implementing the trait.
    ///
    /// This is useful for cases where coherence rules prevents implementing the trait
    /// without adding a dependency to this crate in a third-party library.
    pub unsafe fn new_assert_stable_address(o: O) -> Self
        where O: Deref<Target = T>,
    {
        OwningRef {
            reference: &*o,
            owner: o,
        }
    }

    /// Converts `self` into a new owning reference that points at something reachable
    /// from the previous one.
    ///
    /// This can be a reference to a field of `U`, something reachable from a field of
    /// `U`, or even something unrelated with a `'static` lifetime.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRef;
    ///
    /// fn main() {
    ///     let owning_ref = OwningRef::new(Box::new([1, 2, 3, 4]));
    ///
    ///     // create a owning reference that points at the
    ///     // third element of the array.
    ///     let owning_ref = owning_ref.map(|array| &array[2]);
    ///     assert_eq!(*owning_ref, 3);
    /// }
    /// ```
    pub fn map<F, U: ?Sized>(self, f: F) -> OwningRef<O, U>
        where O: StableAddress,
              F: FnOnce(&T) -> &U
    {
        OwningRef {
            reference: f(&self),
            owner: self.owner,
        }
    }

    /// Tries to convert `self` into a new owning reference that points
    /// at something reachable from the previous one.
    ///
    /// This can be a reference to a field of `U`, something reachable from a field of
    /// `U`, or even something unrelated with a `'static` lifetime.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRef;
    ///
    /// fn main() {
    ///     let owning_ref = OwningRef::new(Box::new([1, 2, 3, 4]));
    ///
    ///     // create a owning reference that points at the
    ///     // third element of the array.
    ///     let owning_ref = owning_ref.try_map(|array| {
    ///         if array[2] == 3 { Ok(&array[2]) } else { Err(()) }
    ///     });
    ///     assert_eq!(*owning_ref.unwrap(), 3);
    /// }
    /// ```
    pub fn try_map<F, U: ?Sized, E>(self, f: F) -> Result<OwningRef<O, U>, E>
        where O: StableAddress,
              F: FnOnce(&T) -> Result<&U, E>
    {
        Ok(OwningRef {
            reference: f(&self)?,
            owner: self.owner,
        })
    }

    /// Converts `self` into a new owning reference with a different owner type.
    ///
    /// The new owner type needs to still contain the original owner in some way
    /// so that the reference into it remains valid. This function is marked unsafe
    /// because the user needs to manually uphold this guarantee.
    pub unsafe fn map_owner<F, P>(self, f: F) -> OwningRef<P, T>
        where O: StableAddress,
              P: StableAddress,
              F: FnOnce(O) -> P
    {
        OwningRef {
            reference: self.reference,
            owner: f(self.owner),
        }
    }

    /// Converts `self` into a new owning reference where the owner is wrapped
    /// in an additional `Box<O>`.
    ///
    /// This can be used to safely erase the owner of any `OwningRef<O, T>`
    /// to a `OwningRef<Box<Erased>, T>`.
    pub fn map_owner_box(self) -> OwningRef<Box<O>, T> {
        OwningRef {
            reference: self.reference,
            owner: Box::new(self.owner),
        }
    }

    /// Erases the concrete base type of the owner with a trait object.
    ///
    /// This allows mixing of owned references with different owner base types.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::{OwningRef, Erased};
    ///
    /// fn main() {
    ///     // NB: Using the concrete types here for explicitnes.
    ///     // For less verbose code type aliases like `BoxRef` are provided.
    ///
    ///     let owning_ref_a: OwningRef<Box<[i32; 4]>, [i32; 4]>
    ///         = OwningRef::new(Box::new([1, 2, 3, 4]));
    ///
    ///     let owning_ref_b: OwningRef<Box<Vec<(i32, bool)>>, Vec<(i32, bool)>>
    ///         = OwningRef::new(Box::new(vec![(0, false), (1, true)]));
    ///
    ///     let owning_ref_a: OwningRef<Box<[i32; 4]>, i32>
    ///         = owning_ref_a.map(|a| &a[0]);
    ///
    ///     let owning_ref_b: OwningRef<Box<Vec<(i32, bool)>>, i32>
    ///         = owning_ref_b.map(|a| &a[1].0);
    ///
    ///     let owning_refs: [OwningRef<Box<Erased>, i32>; 2]
    ///         = [owning_ref_a.erase_owner(), owning_ref_b.erase_owner()];
    ///
    ///     assert_eq!(*owning_refs[0], 1);
    ///     assert_eq!(*owning_refs[1], 1);
    /// }
    /// ```
    pub fn erase_owner<'a>(self) -> OwningRef<O::Erased, T>
        where O: IntoErased<'a>,
    {
        OwningRef {
            reference: self.reference,
            owner: self.owner.into_erased(),
        }
    }

    /// Erases the concrete base type of the owner with a trait object which implements `Send`.
    ///
    /// This allows mixing of owned references with different owner base types.
    pub fn erase_send_owner<'a>(self) -> OwningRef<O::Erased, T>
        where O: IntoErasedSend<'a>,
    {
        OwningRef {
            reference: self.reference,
            owner: self.owner.into_erased_send(),
        }
    }

    /// Erases the concrete base type of the owner with a trait object which implements `Send` and `Sync`.
    ///
    /// This allows mixing of owned references with different owner base types.
    pub fn erase_send_sync_owner<'a>(self) -> OwningRef<O::Erased, T>
        where O: IntoErasedSendSync<'a>,
    {
        OwningRef {
            reference: self.reference,
            owner: self.owner.into_erased_send_sync(),
        }
    }

    // TODO: wrap_owner

    // FIXME: Naming convention?
    /// A getter for the underlying owner.
    pub fn owner(&self) -> &O {
        &self.owner
    }

    // FIXME: Naming convention?
    /// Discards the reference and retrieves the owner.
    pub fn into_inner(self) -> O {
        self.owner
    }
}

impl<O, T: ?Sized> OwningRefMut<O, T> {
    /// Creates a new owning reference from a owner
    /// initialized to the direct dereference of it.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRefMut;
    ///
    /// fn main() {
    ///     let owning_ref_mut = OwningRefMut::new(Box::new(42));
    ///     assert_eq!(*owning_ref_mut, 42);
    /// }
    /// ```
    pub fn new(mut o: O) -> Self
        where O: StableAddress,
              O: DerefMut<Target = T>,
    {
        OwningRefMut {
            reference: &mut *o,
            owner: o,
        }
    }

    /// Like `new`, but doesn’t require `O` to implement the `StableAddress` trait.
    /// Instead, the caller is responsible to make the same promises as implementing the trait.
    ///
    /// This is useful for cases where coherence rules prevents implementing the trait
    /// without adding a dependency to this crate in a third-party library.
    pub unsafe fn new_assert_stable_address(mut o: O) -> Self
        where O: DerefMut<Target = T>,
    {
        OwningRefMut {
            reference: &mut *o,
            owner: o,
        }
    }

    /// Converts `self` into a new _shared_ owning reference that points at
    /// something reachable from the previous one.
    ///
    /// This can be a reference to a field of `U`, something reachable from a field of
    /// `U`, or even something unrelated with a `'static` lifetime.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRefMut;
    ///
    /// fn main() {
    ///     let owning_ref_mut = OwningRefMut::new(Box::new([1, 2, 3, 4]));
    ///
    ///     // create a owning reference that points at the
    ///     // third element of the array.
    ///     let owning_ref = owning_ref_mut.map(|array| &array[2]);
    ///     assert_eq!(*owning_ref, 3);
    /// }
    /// ```
    pub fn map<F, U: ?Sized>(mut self, f: F) -> OwningRef<O, U>
        where O: StableAddress,
              F: FnOnce(&mut T) -> &U
    {
        OwningRef {
            reference: f(&mut self),
            owner: self.owner,
        }
    }

    /// Converts `self` into a new _mutable_ owning reference that points at
    /// something reachable from the previous one.
    ///
    /// This can be a reference to a field of `U`, something reachable from a field of
    /// `U`, or even something unrelated with a `'static` lifetime.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRefMut;
    ///
    /// fn main() {
    ///     let owning_ref_mut = OwningRefMut::new(Box::new([1, 2, 3, 4]));
    ///
    ///     // create a owning reference that points at the
    ///     // third element of the array.
    ///     let owning_ref_mut = owning_ref_mut.map_mut(|array| &mut array[2]);
    ///     assert_eq!(*owning_ref_mut, 3);
    /// }
    /// ```
    pub fn map_mut<F, U: ?Sized>(mut self, f: F) -> OwningRefMut<O, U>
        where O: StableAddress,
              F: FnOnce(&mut T) -> &mut U
    {
        OwningRefMut {
            reference: f(&mut self),
            owner: self.owner,
        }
    }

    /// Tries to convert `self` into a new _shared_ owning reference that points
    /// at something reachable from the previous one.
    ///
    /// This can be a reference to a field of `U`, something reachable from a field of
    /// `U`, or even something unrelated with a `'static` lifetime.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRefMut;
    ///
    /// fn main() {
    ///     let owning_ref_mut = OwningRefMut::new(Box::new([1, 2, 3, 4]));
    ///
    ///     // create a owning reference that points at the
    ///     // third element of the array.
    ///     let owning_ref = owning_ref_mut.try_map(|array| {
    ///         if array[2] == 3 { Ok(&array[2]) } else { Err(()) }
    ///     });
    ///     assert_eq!(*owning_ref.unwrap(), 3);
    /// }
    /// ```
    pub fn try_map<F, U: ?Sized, E>(mut self, f: F) -> Result<OwningRef<O, U>, E>
        where O: StableAddress,
              F: FnOnce(&mut T) -> Result<&U, E>
    {
        Ok(OwningRef {
            reference: f(&mut self)?,
            owner: self.owner,
        })
    }

    /// Tries to convert `self` into a new _mutable_ owning reference that points
    /// at something reachable from the previous one.
    ///
    /// This can be a reference to a field of `U`, something reachable from a field of
    /// `U`, or even something unrelated with a `'static` lifetime.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::OwningRefMut;
    ///
    /// fn main() {
    ///     let owning_ref_mut = OwningRefMut::new(Box::new([1, 2, 3, 4]));
    ///
    ///     // create a owning reference that points at the
    ///     // third element of the array.
    ///     let owning_ref_mut = owning_ref_mut.try_map_mut(|array| {
    ///         if array[2] == 3 { Ok(&mut array[2]) } else { Err(()) }
    ///     });
    ///     assert_eq!(*owning_ref_mut.unwrap(), 3);
    /// }
    /// ```
    pub fn try_map_mut<F, U: ?Sized, E>(mut self, f: F) -> Result<OwningRefMut<O, U>, E>
        where O: StableAddress,
              F: FnOnce(&mut T) -> Result<&mut U, E>
    {
        Ok(OwningRefMut {
            reference: f(&mut self)?,
            owner: self.owner,
        })
    }

    /// Converts `self` into a new owning reference with a different owner type.
    ///
    /// The new owner type needs to still contain the original owner in some way
    /// so that the reference into it remains valid. This function is marked unsafe
    /// because the user needs to manually uphold this guarantee.
    pub unsafe fn map_owner<F, P>(self, f: F) -> OwningRefMut<P, T>
        where O: StableAddress,
              P: StableAddress,
              F: FnOnce(O) -> P
    {
        OwningRefMut {
            reference: self.reference,
            owner: f(self.owner),
        }
    }

    /// Converts `self` into a new owning reference where the owner is wrapped
    /// in an additional `Box<O>`.
    ///
    /// This can be used to safely erase the owner of any `OwningRefMut<O, T>`
    /// to a `OwningRefMut<Box<Erased>, T>`.
    pub fn map_owner_box(self) -> OwningRefMut<Box<O>, T> {
        OwningRefMut {
            reference: self.reference,
            owner: Box::new(self.owner),
        }
    }

    /// Erases the concrete base type of the owner with a trait object.
    ///
    /// This allows mixing of owned references with different owner base types.
    ///
    /// # Example
    /// ```
    /// extern crate owning_ref;
    /// use owning_ref::{OwningRefMut, Erased};
    ///
    /// fn main() {
    ///     // NB: Using the concrete types here for explicitnes.
    ///     // For less verbose code type aliases like `BoxRef` are provided.
    ///
    ///     let owning_ref_mut_a: OwningRefMut<Box<[i32; 4]>, [i32; 4]>
    ///         = OwningRefMut::new(Box::new([1, 2, 3, 4]));
    ///
    ///     let owning_ref_mut_b: OwningRefMut<Box<Vec<(i32, bool)>>, Vec<(i32, bool)>>
    ///         = OwningRefMut::new(Box::new(vec![(0, false), (1, true)]));
    ///
    ///     let owning_ref_mut_a: OwningRefMut<Box<[i32; 4]>, i32>
    ///         = owning_ref_mut_a.map_mut(|a| &mut a[0]);
    ///
    ///     let owning_ref_mut_b: OwningRefMut<Box<Vec<(i32, bool)>>, i32>
    ///         = owning_ref_mut_b.map_mut(|a| &mut a[1].0);
    ///
    ///     let owning_refs_mut: [OwningRefMut<Box<Erased>, i32>; 2]
    ///         = [owning_ref_mut_a.erase_owner(), owning_ref_mut_b.erase_owner()];
    ///
    ///     assert_eq!(*owning_refs_mut[0], 1);
    ///     assert_eq!(*owning_refs_mut[1], 1);
    /// }
    /// ```
    pub fn erase_owner<'a>(self) -> OwningRefMut<O::Erased, T>
        where O: IntoErased<'a>,
    {
        OwningRefMut {
            reference: self.reference,
            owner: self.owner.into_erased(),
        }
    }

    // TODO: wrap_owner

    // FIXME: Naming convention?
    /// A getter for the underlying owner.
    pub fn owner(&self) -> &O {
        &self.owner
    }

    // FIXME: Naming convention?
    /// Discards the reference and retrieves the owner.
    pub fn into_inner(self) -> O {
        self.owner
    }
}

/////////////////////////////////////////////////////////////////////////////
// OwningHandle
/////////////////////////////////////////////////////////////////////////////

use std::ops::{Deref, DerefMut};

/// `OwningHandle` is a complement to `OwningRef`. Where `OwningRef` allows
/// consumers to pass around an owned object and a dependent reference,
/// `OwningHandle` contains an owned object and a dependent _object_.
///
/// `OwningHandle` can encapsulate a `RefMut` along with its associated
/// `RefCell`, or an `RwLockReadGuard` along with its associated `RwLock`.
/// However, the API is completely generic and there are no restrictions on
/// what types of owning and dependent objects may be used.
///
/// `OwningHandle` is created by passing an owner object (which dereferences
/// to a stable address) along with a callback which receives a pointer to
/// that stable location. The callback may then dereference the pointer and
/// mint a dependent object, with the guarantee that the returned object will
/// not outlive the referent of the pointer.
///
/// Since the callback needs to dereference a raw pointer, it requires `unsafe`
/// code. To avoid forcing this unsafety on most callers, the `ToHandle` trait is
/// implemented for common data structures. Types that implement `ToHandle` can
/// be wrapped into an `OwningHandle` without passing a callback.
pub struct OwningHandle<O, H>
    where O: StableAddress, H: Deref,
{
    handle: H,
    _owner: O,
}

impl<O, H> Deref for OwningHandle<O, H>
    where O: StableAddress, H: Deref,
{
    type Target = H::Target;
    fn deref(&self) -> &H::Target {
        self.handle.deref()
    }
}

unsafe impl<O, H> StableAddress for OwningHandle<O, H>
    where O: StableAddress, H: StableAddress,
{}

impl<O, H> DerefMut for OwningHandle<O, H>
    where O: StableAddress, H: DerefMut,
{
    fn deref_mut(&mut self) -> &mut H::Target {
        self.handle.deref_mut()
    }
}

/// Trait to implement the conversion of owner to handle for common types.
pub trait ToHandle {
    /// The type of handle to be encapsulated by the OwningHandle.
    type Handle: Deref;

    /// Given an appropriately-long-lived pointer to ourselves, create a
    /// handle to be encapsulated by the `OwningHandle`.
    unsafe fn to_handle(x: *const Self) -> Self::Handle;
}

/// Trait to implement the conversion of owner to mutable handle for common types.
pub trait ToHandleMut {
    /// The type of handle to be encapsulated by the OwningHandle.
    type HandleMut: DerefMut;

    /// Given an appropriately-long-lived pointer to ourselves, create a
    /// mutable handle to be encapsulated by the `OwningHandle`.
    unsafe fn to_handle_mut(x: *const Self) -> Self::HandleMut;
}

impl<O, H> OwningHandle<O, H>
    where O: StableAddress, O::Target: ToHandle<Handle = H>, H: Deref,
{
    /// Create a new `OwningHandle` for a type that implements `ToHandle`. For types
    /// that don't implement `ToHandle`, callers may invoke `new_with_fn`, which accepts
    /// a callback to perform the conversion.
    pub fn new(o: O) -> Self {
        OwningHandle::new_with_fn(o, |x| unsafe { O::Target::to_handle(x) })
    }
}

impl<O, H> OwningHandle<O, H>
    where O: StableAddress, O::Target: ToHandleMut<HandleMut = H>, H: DerefMut,
{
    /// Create a new mutable `OwningHandle` for a type that implements `ToHandleMut`.
    pub fn new_mut(o: O) -> Self {
        OwningHandle::new_with_fn(o, |x| unsafe { O::Target::to_handle_mut(x) })
    }
}

impl<O, H> OwningHandle<O, H>
    where O: StableAddress, H: Deref,
{
    /// Create a new OwningHandle. The provided callback will be invoked with
    /// a pointer to the object owned by `o`, and the returned value is stored
    /// as the object to which this `OwningHandle` will forward `Deref` and
    /// `DerefMut`.
    pub fn new_with_fn<F>(o: O, f: F) -> Self
        where F: FnOnce(*const O::Target) -> H
    {
        let h: H;
        {
            h = f(o.deref() as *const O::Target);
        }

        OwningHandle {
          handle: h,
          _owner: o,
        }
    }

    /// Create a new OwningHandle. The provided callback will be invoked with
    /// a pointer to the object owned by `o`, and the returned value is stored
    /// as the object to which this `OwningHandle` will forward `Deref` and
    /// `DerefMut`.
    pub fn try_new<F, E>(o: O, f: F) -> Result<Self, E>
        where F: FnOnce(*const O::Target) -> Result<H, E>
    {
        let h: H;
        {
            h = f(o.deref() as *const O::Target)?;
        }

        Ok(OwningHandle {
          handle: h,
          _owner: o,
        })
    }
}

/////////////////////////////////////////////////////////////////////////////
// std traits
/////////////////////////////////////////////////////////////////////////////

use std::convert::From;
use std::fmt::{self, Debug};
use std::marker::{Send, Sync};
use std::cmp::{Eq, PartialEq, Ord, PartialOrd, Ordering};
use std::hash::{Hash, Hasher};
use std::borrow::Borrow;

impl<O, T: ?Sized> Deref for OwningRef<O, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self.reference
        }
    }
}

impl<O, T: ?Sized> Deref for OwningRefMut<O, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self.reference
        }
    }
}

impl<O, T: ?Sized> DerefMut for OwningRefMut<O, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            &mut *self.reference
        }
    }
}

unsafe impl<O, T: ?Sized> StableAddress for OwningRef<O, T> {}

impl<O, T: ?Sized> AsRef<T> for OwningRef<O, T> {
    fn as_ref(&self) -> &T {
        &*self
    }
}

impl<O, T: ?Sized> AsRef<T> for OwningRefMut<O, T> {
    fn as_ref(&self) -> &T {
        &*self
    }
}

impl<O, T: ?Sized> AsMut<T> for OwningRefMut<O, T> {
    fn as_mut(&mut self) -> &mut T {
        &mut *self
    }
}

impl<O, T: ?Sized> Borrow<T> for OwningRef<O, T> {
    fn borrow(&self) -> &T {
        &*self
    }
}

impl<O, T: ?Sized> From<O> for OwningRef<O, T>
    where O: StableAddress,
          O: Deref<Target = T>,
{
    fn from(owner: O) -> Self {
        OwningRef::new(owner)
    }
}

impl<O, T: ?Sized> From<O> for OwningRefMut<O, T>
    where O: StableAddress,
          O: DerefMut<Target = T>
{
    fn from(owner: O) -> Self {
        OwningRefMut::new(owner)
    }
}

impl<O, T: ?Sized> From<OwningRefMut<O, T>> for OwningRef<O, T>
    where O: StableAddress,
          O: DerefMut<Target = T>
{
    fn from(other: OwningRefMut<O, T>) -> Self {
        OwningRef {
            owner: other.owner,
            reference: other.reference,
        }
    }
}

// ^ FIXME: Is a Into impl for calling into_inner() possible as well?

impl<O, T: ?Sized> Debug for OwningRef<O, T>
    where O: Debug,
          T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "OwningRef {{ owner: {:?}, reference: {:?} }}",
               self.owner(),
               &**self)
    }
}

impl<O, T: ?Sized> Debug for OwningRefMut<O, T>
    where O: Debug,
          T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "OwningRefMut {{ owner: {:?}, reference: {:?} }}",
               self.owner(),
               &**self)
    }
}

impl<O, T: ?Sized> Clone for OwningRef<O, T>
    where O: CloneStableAddress,
{
    fn clone(&self) -> Self {
        OwningRef {
            owner: self.owner.clone(),
            reference: self.reference,
        }
    }
}

unsafe impl<O, T: ?Sized> CloneStableAddress for OwningRef<O, T>
    where O: CloneStableAddress {}

unsafe impl<O, T: ?Sized> Send for OwningRef<O, T>
    where O: Send, for<'a> (&'a T): Send {}
unsafe impl<O, T: ?Sized> Sync for OwningRef<O, T>
    where O: Sync, for<'a> (&'a T): Sync {}

unsafe impl<O, T: ?Sized> Send for OwningRefMut<O, T>
    where O: Send, for<'a> (&'a mut T): Send {}
unsafe impl<O, T: ?Sized> Sync for OwningRefMut<O, T>
    where O: Sync, for<'a> (&'a mut T): Sync {}

impl Debug for dyn Erased {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<Erased>",)
    }
}

impl<O, T: ?Sized> PartialEq for OwningRef<O, T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        (&*self as &T).eq(&*other as &T)
     }
}

impl<O, T: ?Sized> Eq for OwningRef<O, T> where T: Eq {}

impl<O, T: ?Sized> PartialOrd for OwningRef<O, T> where T: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&*self as &T).partial_cmp(&*other as &T)
    }
}

impl<O, T: ?Sized> Ord for OwningRef<O, T> where T: Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        (&*self as &T).cmp(&*other as &T)
    }
}

impl<O, T: ?Sized> Hash for OwningRef<O, T> where T: Hash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&*self as &T).hash(state);
    }
}

impl<O, T: ?Sized> PartialEq for OwningRefMut<O, T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        (&*self as &T).eq(&*other as &T)
     }
}

impl<O, T: ?Sized> Eq for OwningRefMut<O, T> where T: Eq {}

impl<O, T: ?Sized> PartialOrd for OwningRefMut<O, T> where T: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&*self as &T).partial_cmp(&*other as &T)
    }
}

impl<O, T: ?Sized> Ord for OwningRefMut<O, T> where T: Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        (&*self as &T).cmp(&*other as &T)
    }
}

impl<O, T: ?Sized> Hash for OwningRefMut<O, T> where T: Hash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&*self as &T).hash(state);
    }
}

/////////////////////////////////////////////////////////////////////////////
// std types integration and convenience type defs
/////////////////////////////////////////////////////////////////////////////

use std::boxed::Box;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::{MutexGuard, RwLockReadGuard, RwLockWriteGuard};
use std::cell::{Ref, RefCell, RefMut};

impl<T: 'static> ToHandle for RefCell<T> {
    type Handle = Ref<'static, T>;
    unsafe fn to_handle(x: *const Self) -> Self::Handle { (*x).borrow() }
}

impl<T: 'static> ToHandleMut for RefCell<T> {
    type HandleMut = RefMut<'static, T>;
    unsafe fn to_handle_mut(x: *const Self) -> Self::HandleMut { (*x).borrow_mut() }
}

// NB: Implementing ToHandle{,Mut} for Mutex and RwLock requires a decision
// about which handle creation to use (i.e. read() vs try_read()) as well as
// what to do with error results.

/// Typedef of a owning reference that uses a `Box` as the owner.
pub type BoxRef<T, U = T> = OwningRef<Box<T>, U>;
/// Typedef of a owning reference that uses a `Vec` as the owner.
pub type VecRef<T, U = T> = OwningRef<Vec<T>, U>;
/// Typedef of a owning reference that uses a `String` as the owner.
pub type StringRef = OwningRef<String, str>;

/// Typedef of a owning reference that uses a `Rc` as the owner.
pub type RcRef<T, U = T> = OwningRef<Rc<T>, U>;
/// Typedef of a owning reference that uses a `Arc` as the owner.
pub type ArcRef<T, U = T> = OwningRef<Arc<T>, U>;

/// Typedef of a owning reference that uses a `Ref` as the owner.
pub type RefRef<'a, T, U = T> = OwningRef<Ref<'a, T>, U>;
/// Typedef of a owning reference that uses a `RefMut` as the owner.
pub type RefMutRef<'a, T, U = T> = OwningRef<RefMut<'a, T>, U>;
/// Typedef of a owning reference that uses a `MutexGuard` as the owner.
pub type MutexGuardRef<'a, T, U = T> = OwningRef<MutexGuard<'a, T>, U>;
/// Typedef of a owning reference that uses a `RwLockReadGuard` as the owner.
pub type RwLockReadGuardRef<'a, T, U = T> = OwningRef<RwLockReadGuard<'a, T>, U>;
/// Typedef of a owning reference that uses a `RwLockWriteGuard` as the owner.
pub type RwLockWriteGuardRef<'a, T, U = T> = OwningRef<RwLockWriteGuard<'a, T>, U>;

/// Typedef of a mutable owning reference that uses a `Box` as the owner.
pub type BoxRefMut<T, U = T> = OwningRefMut<Box<T>, U>;
/// Typedef of a mutable owning reference that uses a `Vec` as the owner.
pub type VecRefMut<T, U = T> = OwningRefMut<Vec<T>, U>;
/// Typedef of a mutable owning reference that uses a `String` as the owner.
pub type StringRefMut = OwningRefMut<String, str>;

/// Typedef of a mutable owning reference that uses a `RefMut` as the owner.
pub type RefMutRefMut<'a, T, U = T> = OwningRefMut<RefMut<'a, T>, U>;
/// Typedef of a mutable owning reference that uses a `MutexGuard` as the owner.
pub type MutexGuardRefMut<'a, T, U = T> = OwningRefMut<MutexGuard<'a, T>, U>;
/// Typedef of a mutable owning reference that uses a `RwLockWriteGuard` as the owner.
pub type RwLockWriteGuardRefMut<'a, T, U = T> = OwningRef<RwLockWriteGuard<'a, T>, U>;

unsafe impl<'a, T: 'a> IntoErased<'a> for Box<T> {
    type Erased = Box<dyn Erased + 'a>;
    fn into_erased(self) -> Self::Erased {
        self
    }
}
unsafe impl<'a, T: 'a> IntoErased<'a> for Rc<T> {
    type Erased = Rc<dyn Erased + 'a>;
    fn into_erased(self) -> Self::Erased {
        self
    }
}
unsafe impl<'a, T: 'a> IntoErased<'a> for Arc<T> {
    type Erased = Arc<dyn Erased + 'a>;
    fn into_erased(self) -> Self::Erased {
        self
    }
}

unsafe impl<'a, T: Send + 'a> IntoErasedSend<'a> for Box<T> {
    type Erased = Box<dyn Erased + Send + 'a>;
    fn into_erased_send(self) -> Self::Erased {
        self
    }
}

unsafe impl<'a, T: Send + 'a> IntoErasedSendSync<'a> for Box<T> {
    type Erased = Box<dyn Erased + Sync + Send + 'a>;
    fn into_erased_send_sync(self) -> Self::Erased {
        let result: Box<dyn Erased + Send + 'a> = self;
        // This is safe since Erased can always implement Sync
        // Only the destructor is available and it takes &mut self
        unsafe {
            mem::transmute(result)
        }
    }
}

unsafe impl<'a, T: Send + Sync + 'a> IntoErasedSendSync<'a> for Arc<T> {
    type Erased = Arc<dyn Erased + Send + Sync + 'a>;
    fn into_erased_send_sync(self) -> Self::Erased {
        self
    }
}

/// Typedef of a owning reference that uses an erased `Box` as the owner.
pub type ErasedBoxRef<U> = OwningRef<Box<dyn Erased>, U>;
/// Typedef of a owning reference that uses an erased `Rc` as the owner.
pub type ErasedRcRef<U> = OwningRef<Rc<dyn Erased>, U>;
/// Typedef of a owning reference that uses an erased `Arc` as the owner.
pub type ErasedArcRef<U> = OwningRef<Arc<dyn Erased>, U>;

/// Typedef of a mutable owning reference that uses an erased `Box` as the owner.
pub type ErasedBoxRefMut<U> = OwningRefMut<Box<dyn Erased>, U>;

#[cfg(test)]
mod tests {
    mod owning_ref {
        use super::super::OwningRef;
        use super::super::{RcRef, BoxRef, Erased, ErasedBoxRef};
        use std::cmp::{PartialEq, Ord, PartialOrd, Ordering};
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        use std::collections::HashMap;
        use std::rc::Rc;

        #[derive(Debug, PartialEq)]
        struct Example(u32, String, [u8; 3]);
        fn example() -> Example {
            Example(42, "hello world".to_string(), [1, 2, 3])
        }

        #[test]
        fn new_deref() {
            let or: OwningRef<Box<()>, ()> = OwningRef::new(Box::new(()));
            assert_eq!(&*or, &());
        }

        #[test]
        fn into() {
            let or: OwningRef<Box<()>, ()> = Box::new(()).into();
            assert_eq!(&*or, &());
        }

        #[test]
        fn map_offset_ref() {
            let or: BoxRef<Example> = Box::new(example()).into();
            let or: BoxRef<_, u32> = or.map(|x| &x.0);
            assert_eq!(&*or, &42);

            let or: BoxRef<Example> = Box::new(example()).into();
            let or: BoxRef<_, u8> = or.map(|x| &x.2[1]);
            assert_eq!(&*or, &2);
        }

        #[test]
        fn map_heap_ref() {
            let or: BoxRef<Example> = Box::new(example()).into();
            let or: BoxRef<_, str> = or.map(|x| &x.1[..5]);
            assert_eq!(&*or, "hello");
        }

        #[test]
        fn map_static_ref() {
            let or: BoxRef<()> = Box::new(()).into();
            let or: BoxRef<_, str> = or.map(|_| "hello");
            assert_eq!(&*or, "hello");
        }

        #[test]
        fn map_chained() {
            let or: BoxRef<String> = Box::new(example().1).into();
            let or: BoxRef<_, str> = or.map(|x| &x[1..5]);
            let or: BoxRef<_, str> = or.map(|x| &x[..2]);
            assert_eq!(&*or, "el");
        }

        #[test]
        fn map_chained_inference() {
            let or = BoxRef::new(Box::new(example().1))
                .map(|x| &x[..5])
                .map(|x| &x[1..3]);
            assert_eq!(&*or, "el");
        }

        #[test]
        fn owner() {
            let or: BoxRef<String> = Box::new(example().1).into();
            let or = or.map(|x| &x[..5]);
            assert_eq!(&*or, "hello");
            assert_eq!(&**or.owner(), "hello world");
        }

        #[test]
        fn into_inner() {
            let or: BoxRef<String> = Box::new(example().1).into();
            let or = or.map(|x| &x[..5]);
            assert_eq!(&*or, "hello");
            let s = *or.into_inner();
            assert_eq!(&s, "hello world");
        }

        #[test]
        fn fmt_debug() {
            let or: BoxRef<String> = Box::new(example().1).into();
            let or = or.map(|x| &x[..5]);
            let s = format!("{:?}", or);
            assert_eq!(&s, "OwningRef { owner: \"hello world\", reference: \"hello\" }");
        }

        #[test]
        fn erased_owner() {
            let o1: BoxRef<Example, str> = BoxRef::new(Box::new(example()))
                .map(|x| &x.1[..]);

            let o2: BoxRef<String, str> = BoxRef::new(Box::new(example().1))
                .map(|x| &x[..]);

            let os: Vec<ErasedBoxRef<str>> = vec![o1.erase_owner(), o2.erase_owner()];
            assert!(os.iter().all(|e| &e[..] == "hello world"));
        }

        #[test]
        fn raii_locks() {
            use super::super::{RefRef, RefMutRef};
            use std::cell::RefCell;
            use super::super::{MutexGuardRef, RwLockReadGuardRef, RwLockWriteGuardRef};
            use std::sync::{Mutex, RwLock};

            {
                let a = RefCell::new(1);
                let a = {
                    let a = RefRef::new(a.borrow());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
            {
                let a = RefCell::new(1);
                let a = {
                    let a = RefMutRef::new(a.borrow_mut());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
            {
                let a = Mutex::new(1);
                let a = {
                    let a = MutexGuardRef::new(a.lock().unwrap());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
            {
                let a = RwLock::new(1);
                let a = {
                    let a = RwLockReadGuardRef::new(a.read().unwrap());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
            {
                let a = RwLock::new(1);
                let a = {
                    let a = RwLockWriteGuardRef::new(a.write().unwrap());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
        }

        #[test]
        fn eq() {
            let or1: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
            let or2: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
            assert_eq!(or1.eq(&or2), true);
        }

        #[test]
        fn cmp() {
            let or1: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
            let or2: BoxRef<[u8]> = BoxRef::new(vec![4, 5, 6].into_boxed_slice());
            assert_eq!(or1.cmp(&or2), Ordering::Less);
        }

        #[test]
        fn partial_cmp() {
            let or1: BoxRef<[u8]> = BoxRef::new(vec![4, 5, 6].into_boxed_slice());
            let or2: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
            assert_eq!(or1.partial_cmp(&or2), Some(Ordering::Greater));
        }

        #[test]
        fn hash() {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();

            let or1: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());
            let or2: BoxRef<[u8]> = BoxRef::new(vec![1, 2, 3].into_boxed_slice());

            or1.hash(&mut h1);
            or2.hash(&mut h2);

            assert_eq!(h1.finish(), h2.finish());
        }

        #[test]
        fn borrow() {
            let mut hash = HashMap::new();
            let     key  = RcRef::<String>::new(Rc::new("foo-bar".to_string())).map(|s| &s[..]);

            hash.insert(key.clone().map(|s| &s[..3]), 42);
            hash.insert(key.clone().map(|s| &s[4..]), 23);

            assert_eq!(hash.get("foo"), Some(&42));
            assert_eq!(hash.get("bar"), Some(&23));
        }

        #[test]
        fn total_erase() {
            let a: OwningRef<Vec<u8>, [u8]>
                = OwningRef::new(vec![]).map(|x| &x[..]);
            let b: OwningRef<Box<[u8]>, [u8]>
                = OwningRef::new(vec![].into_boxed_slice()).map(|x| &x[..]);

            let c: OwningRef<Rc<Vec<u8>>, [u8]> = unsafe {a.map_owner(Rc::new)};
            let d: OwningRef<Rc<Box<[u8]>>, [u8]> = unsafe {b.map_owner(Rc::new)};

            let e: OwningRef<Rc<dyn Erased>, [u8]> = c.erase_owner();
            let f: OwningRef<Rc<dyn Erased>, [u8]> = d.erase_owner();

            let _g = e.clone();
            let _h = f.clone();
        }

        #[test]
        fn total_erase_box() {
            let a: OwningRef<Vec<u8>, [u8]>
                = OwningRef::new(vec![]).map(|x| &x[..]);
            let b: OwningRef<Box<[u8]>, [u8]>
                = OwningRef::new(vec![].into_boxed_slice()).map(|x| &x[..]);

            let c: OwningRef<Box<Vec<u8>>, [u8]> = a.map_owner_box();
            let d: OwningRef<Box<Box<[u8]>>, [u8]> = b.map_owner_box();

            let _e: OwningRef<Box<dyn Erased>, [u8]> = c.erase_owner();
            let _f: OwningRef<Box<dyn Erased>, [u8]> = d.erase_owner();
        }

        #[test]
        fn try_map1() {
            use std::any::Any;

            let x = Box::new(123_i32);
            let y: Box<dyn Any> = x;

            OwningRef::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_ok();
        }

        #[test]
        fn try_map2() {
            use std::any::Any;

            let x = Box::new(123_i32);
            let y: Box<dyn Any> = x;

            OwningRef::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_err();
        }
    }

    mod owning_handle {
        use super::super::OwningHandle;
        use super::super::RcRef;
        use std::rc::Rc;
        use std::cell::RefCell;
        use std::sync::Arc;
        use std::sync::RwLock;

        #[test]
        fn owning_handle() {
            use std::cell::RefCell;
            let cell = Rc::new(RefCell::new(2));
            let cell_ref = RcRef::new(cell);
            let mut handle = OwningHandle::new_with_fn(cell_ref, |x| unsafe { x.as_ref() }.unwrap().borrow_mut());
            assert_eq!(*handle, 2);
            *handle = 3;
            assert_eq!(*handle, 3);
        }

        #[test]
        fn try_owning_handle_ok() {
            use std::cell::RefCell;
            let cell = Rc::new(RefCell::new(2));
            let cell_ref = RcRef::new(cell);
            let mut handle = OwningHandle::try_new::<_, ()>(cell_ref, |x| {
                Ok(unsafe {
                    x.as_ref()
                }.unwrap().borrow_mut())
            }).unwrap();
            assert_eq!(*handle, 2);
            *handle = 3;
            assert_eq!(*handle, 3);
        }

        #[test]
        fn try_owning_handle_err() {
            use std::cell::RefCell;
            let cell = Rc::new(RefCell::new(2));
            let cell_ref = RcRef::new(cell);
            let handle = OwningHandle::try_new::<_, ()>(cell_ref, |x| {
                if false {
                    return Ok(unsafe {
                        x.as_ref()
                    }.unwrap().borrow_mut())
                }
                Err(())
            });
            assert!(handle.is_err());
        }

        #[test]
        fn nested() {
            use std::cell::RefCell;
            use std::sync::{Arc, RwLock};

            let result = {
                let complex = Rc::new(RefCell::new(Arc::new(RwLock::new("someString"))));
                let curr = RcRef::new(complex);
                let curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().borrow_mut());
                let mut curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().try_write().unwrap());
                assert_eq!(*curr, "someString");
                *curr = "someOtherString";
                curr
            };
            assert_eq!(*result, "someOtherString");
        }

        #[test]
        fn owning_handle_safe() {
            use std::cell::RefCell;
            let cell = Rc::new(RefCell::new(2));
            let cell_ref = RcRef::new(cell);
            let handle = OwningHandle::new(cell_ref);
            assert_eq!(*handle, 2);
        }

        #[test]
        fn owning_handle_mut_safe() {
            use std::cell::RefCell;
            let cell = Rc::new(RefCell::new(2));
            let cell_ref = RcRef::new(cell);
            let mut handle = OwningHandle::new_mut(cell_ref);
            assert_eq!(*handle, 2);
            *handle = 3;
            assert_eq!(*handle, 3);
        }

        #[test]
        fn owning_handle_safe_2() {
            let result = {
                let complex = Rc::new(RefCell::new(Arc::new(RwLock::new("someString"))));
                let curr = RcRef::new(complex);
                let curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().borrow_mut());
                let mut curr = OwningHandle::new_with_fn(curr, |x| unsafe { x.as_ref() }.unwrap().try_write().unwrap());
                assert_eq!(*curr, "someString");
                *curr = "someOtherString";
                curr
            };
            assert_eq!(*result, "someOtherString");
        }
    }

    mod owning_ref_mut {
        use super::super::{OwningRefMut, BoxRefMut, Erased, ErasedBoxRefMut};
        use super::super::BoxRef;
        use std::cmp::{PartialEq, Ord, PartialOrd, Ordering};
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        use std::collections::HashMap;

        #[derive(Debug, PartialEq)]
        struct Example(u32, String, [u8; 3]);
        fn example() -> Example {
            Example(42, "hello world".to_string(), [1, 2, 3])
        }

        #[test]
        fn new_deref() {
            let or: OwningRefMut<Box<()>, ()> = OwningRefMut::new(Box::new(()));
            assert_eq!(&*or, &());
        }

        #[test]
        fn new_deref_mut() {
            let mut or: OwningRefMut<Box<()>, ()> = OwningRefMut::new(Box::new(()));
            assert_eq!(&mut *or, &mut ());
        }

        #[test]
        fn mutate() {
            let mut or: OwningRefMut<Box<usize>, usize> = OwningRefMut::new(Box::new(0));
            assert_eq!(&*or, &0);
            *or = 1;
            assert_eq!(&*or, &1);
        }

        #[test]
        fn into() {
            let or: OwningRefMut<Box<()>, ()> = Box::new(()).into();
            assert_eq!(&*or, &());
        }

        #[test]
        fn map_offset_ref() {
            let or: BoxRefMut<Example> = Box::new(example()).into();
            let or: BoxRef<_, u32> = or.map(|x| &mut x.0);
            assert_eq!(&*or, &42);

            let or: BoxRefMut<Example> = Box::new(example()).into();
            let or: BoxRef<_, u8> = or.map(|x| &mut x.2[1]);
            assert_eq!(&*or, &2);
        }

        #[test]
        fn map_heap_ref() {
            let or: BoxRefMut<Example> = Box::new(example()).into();
            let or: BoxRef<_, str> = or.map(|x| &mut x.1[..5]);
            assert_eq!(&*or, "hello");
        }

        #[test]
        fn map_static_ref() {
            let or: BoxRefMut<()> = Box::new(()).into();
            let or: BoxRef<_, str> = or.map(|_| "hello");
            assert_eq!(&*or, "hello");
        }

        #[test]
        fn map_mut_offset_ref() {
            let or: BoxRefMut<Example> = Box::new(example()).into();
            let or: BoxRefMut<_, u32> = or.map_mut(|x| &mut x.0);
            assert_eq!(&*or, &42);

            let or: BoxRefMut<Example> = Box::new(example()).into();
            let or: BoxRefMut<_, u8> = or.map_mut(|x| &mut x.2[1]);
            assert_eq!(&*or, &2);
        }

        #[test]
        fn map_mut_heap_ref() {
            let or: BoxRefMut<Example> = Box::new(example()).into();
            let or: BoxRefMut<_, str> = or.map_mut(|x| &mut x.1[..5]);
            assert_eq!(&*or, "hello");
        }

        #[test]
        fn map_mut_static_ref() {
            static mut MUT_S: [u8; 5] = *b"hello";

            let mut_s: &'static mut [u8] = unsafe { &mut MUT_S };

            let or: BoxRefMut<()> = Box::new(()).into();
            let or: BoxRefMut<_, [u8]> = or.map_mut(move |_| mut_s);
            assert_eq!(&*or, b"hello");
        }

        #[test]
        fn map_mut_chained() {
            let or: BoxRefMut<String> = Box::new(example().1).into();
            let or: BoxRefMut<_, str> = or.map_mut(|x| &mut x[1..5]);
            let or: BoxRefMut<_, str> = or.map_mut(|x| &mut x[..2]);
            assert_eq!(&*or, "el");
        }

        #[test]
        fn map_chained_inference() {
            let or = BoxRefMut::new(Box::new(example().1))
                .map_mut(|x| &mut x[..5])
                .map_mut(|x| &mut x[1..3]);
            assert_eq!(&*or, "el");
        }

        #[test]
        fn try_map_mut() {
            let or: BoxRefMut<String> = Box::new(example().1).into();
            let or: Result<BoxRefMut<_, str>, ()> = or.try_map_mut(|x| Ok(&mut x[1..5]));
            assert_eq!(&*or.unwrap(), "ello");

            let or: BoxRefMut<String> = Box::new(example().1).into();
            let or: Result<BoxRefMut<_, str>, ()> = or.try_map_mut(|_| Err(()));
            assert!(or.is_err());
        }

        #[test]
        fn owner() {
            let or: BoxRefMut<String> = Box::new(example().1).into();
            let or = or.map_mut(|x| &mut x[..5]);
            assert_eq!(&*or, "hello");
            assert_eq!(&**or.owner(), "hello world");
        }

        #[test]
        fn into_inner() {
            let or: BoxRefMut<String> = Box::new(example().1).into();
            let or = or.map_mut(|x| &mut x[..5]);
            assert_eq!(&*or, "hello");
            let s = *or.into_inner();
            assert_eq!(&s, "hello world");
        }

        #[test]
        fn fmt_debug() {
            let or: BoxRefMut<String> = Box::new(example().1).into();
            let or = or.map_mut(|x| &mut x[..5]);
            let s = format!("{:?}", or);
            assert_eq!(&s,
                       "OwningRefMut { owner: \"hello world\", reference: \"hello\" }");
        }

        #[test]
        fn erased_owner() {
            let o1: BoxRefMut<Example, str> = BoxRefMut::new(Box::new(example()))
                .map_mut(|x| &mut x.1[..]);

            let o2: BoxRefMut<String, str> = BoxRefMut::new(Box::new(example().1))
                .map_mut(|x| &mut x[..]);

            let os: Vec<ErasedBoxRefMut<str>> = vec![o1.erase_owner(), o2.erase_owner()];
            assert!(os.iter().all(|e| &e[..] == "hello world"));
        }

        #[test]
        fn raii_locks() {
            use super::super::RefMutRefMut;
            use std::cell::RefCell;
            use super::super::{MutexGuardRefMut, RwLockWriteGuardRefMut};
            use std::sync::{Mutex, RwLock};

            {
                let a = RefCell::new(1);
                let a = {
                    let a = RefMutRefMut::new(a.borrow_mut());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
            {
                let a = Mutex::new(1);
                let a = {
                    let a = MutexGuardRefMut::new(a.lock().unwrap());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
            {
                let a = RwLock::new(1);
                let a = {
                    let a = RwLockWriteGuardRefMut::new(a.write().unwrap());
                    assert_eq!(*a, 1);
                    a
                };
                assert_eq!(*a, 1);
                drop(a);
            }
        }

        #[test]
        fn eq() {
            let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
            let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
            assert_eq!(or1.eq(&or2), true);
        }

        #[test]
        fn cmp() {
            let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
            let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![4, 5, 6].into_boxed_slice());
            assert_eq!(or1.cmp(&or2), Ordering::Less);
        }

        #[test]
        fn partial_cmp() {
            let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![4, 5, 6].into_boxed_slice());
            let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
            assert_eq!(or1.partial_cmp(&or2), Some(Ordering::Greater));
        }

        #[test]
        fn hash() {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();

            let or1: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());
            let or2: BoxRefMut<[u8]> = BoxRefMut::new(vec![1, 2, 3].into_boxed_slice());

            or1.hash(&mut h1);
            or2.hash(&mut h2);

            assert_eq!(h1.finish(), h2.finish());
        }

        #[test]
        fn borrow() {
            let mut hash = HashMap::new();
            let     key1 = BoxRefMut::<String>::new(Box::new("foo".to_string())).map(|s| &s[..]);
            let     key2 = BoxRefMut::<String>::new(Box::new("bar".to_string())).map(|s| &s[..]);

            hash.insert(key1, 42);
            hash.insert(key2, 23);

            assert_eq!(hash.get("foo"), Some(&42));
            assert_eq!(hash.get("bar"), Some(&23));
        }

        #[test]
        fn total_erase() {
            let a: OwningRefMut<Vec<u8>, [u8]>
                = OwningRefMut::new(vec![]).map_mut(|x| &mut x[..]);
            let b: OwningRefMut<Box<[u8]>, [u8]>
                = OwningRefMut::new(vec![].into_boxed_slice()).map_mut(|x| &mut x[..]);

            let c: OwningRefMut<Box<Vec<u8>>, [u8]> = unsafe {a.map_owner(Box::new)};
            let d: OwningRefMut<Box<Box<[u8]>>, [u8]> = unsafe {b.map_owner(Box::new)};

            let _e: OwningRefMut<Box<dyn Erased>, [u8]> = c.erase_owner();
            let _f: OwningRefMut<Box<dyn Erased>, [u8]> = d.erase_owner();
        }

        #[test]
        fn total_erase_box() {
            let a: OwningRefMut<Vec<u8>, [u8]>
                = OwningRefMut::new(vec![]).map_mut(|x| &mut x[..]);
            let b: OwningRefMut<Box<[u8]>, [u8]>
                = OwningRefMut::new(vec![].into_boxed_slice()).map_mut(|x| &mut x[..]);

            let c: OwningRefMut<Box<Vec<u8>>, [u8]> = a.map_owner_box();
            let d: OwningRefMut<Box<Box<[u8]>>, [u8]> = b.map_owner_box();

            let _e: OwningRefMut<Box<dyn Erased>, [u8]> = c.erase_owner();
            let _f: OwningRefMut<Box<dyn Erased>, [u8]> = d.erase_owner();
        }

        #[test]
        fn try_map1() {
            use std::any::Any;

            let x = Box::new(123_i32);
            let y: Box<dyn Any> = x;

            OwningRefMut::new(y).try_map_mut(|x| x.downcast_mut::<i32>().ok_or(())).is_ok();
        }

        #[test]
        fn try_map2() {
            use std::any::Any;

            let x = Box::new(123_i32);
            let y: Box<dyn Any> = x;

            OwningRefMut::new(y).try_map_mut(|x| x.downcast_mut::<i32>().ok_or(())).is_err();
        }

        #[test]
        fn try_map3() {
            use std::any::Any;

            let x = Box::new(123_i32);
            let y: Box<dyn Any> = x;

            OwningRefMut::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_ok();
        }

        #[test]
        fn try_map4() {
            use std::any::Any;

            let x = Box::new(123_i32);
            let y: Box<dyn Any> = x;

            OwningRefMut::new(y).try_map(|x| x.downcast_ref::<i32>().ok_or(())).is_err();
        }

        #[test]
        fn into_owning_ref() {
            use super::super::BoxRef;

            let or: BoxRefMut<()> = Box::new(()).into();
            let or: BoxRef<()> = or.into();
            assert_eq!(&*or, &());
        }

        struct Foo {
            u: u32,
        }
        struct Bar {
            f: Foo,
        }

        #[test]
        fn ref_mut() {
            use std::cell::RefCell;

            let a = RefCell::new(Bar { f: Foo { u: 42 } });
            let mut b = OwningRefMut::new(a.borrow_mut());
            assert_eq!(b.f.u, 42);
            b.f.u = 43;
            let mut c = b.map_mut(|x| &mut x.f);
            assert_eq!(c.u, 43);
            c.u = 44;
            let mut d = c.map_mut(|x| &mut x.u);
            assert_eq!(*d, 44);
            *d = 45;
            assert_eq!(*d, 45);
        }
    }
}
