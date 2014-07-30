// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Task-local reference-counted boxes (`Rc` type)

The `Rc` type provides shared ownership of an immutable value. Destruction is
deterministic, and will occur as soon as the last owner is gone. It is marked
as non-sendable because it avoids the overhead of atomic reference counting.

The `downgrade` method can be used to create a non-owning `Weak` pointer to the
box. A `Weak` pointer can be upgraded to an `Rc` pointer, but will return
`None` if the value has already been freed.

For example, a tree with parent pointers can be represented by putting the
nodes behind strong `Rc` pointers, and then storing the parent pointers as
`Weak` pointers.


## Examples

Consider a scenario where a set of Gadgets are owned by a given Owner.  We want
to have our Gadgets point to their Owner.  We can't do this with unique
ownership, because more than one gadget may belong to the same Owner.  Rc
allows us to share an Owner between multiple Gadgets, and have the Owner kept
alive as long as any Gadget points at it.

```rust
use std::rc::Rc;

struct Owner {
    name: String
    // ...other fields
}

struct Gadget {
    id: int,
    owner: Rc<Owner>
    // ...other fields
}

fn main() {
    // Create a reference counted Owner.
    let gadget_owner : Rc<Owner> = Rc::new(
            Owner { name: String::from_str("Gadget Man") }
    );

    // Create Gadgets belonging to gadget_owner.  To increment the reference
    // count we clone the Rc object.
    let gadget1 = Gadget { id: 1, owner: gadget_owner.clone() };
    let gadget2 = Gadget { id: 2, owner: gadget_owner.clone() };

    drop(gadget_owner);

    // Despite dropping gadget_owner, we're still able to print out the name of
    // the Owner of the Gadgets. This is because we've only dropped the
    // reference count object, not the Owner it wraps. As long as there are
    // other Rc objects pointing at the same Owner, it will stay alive. Notice
    // that the Rc wrapper around Gadget.owner gets automatically dereferenced
    // for us.
    println!("Gadget {} owned by {}", gadget1.id, gadget1.owner.name);
    println!("Gadget {} owned by {}", gadget2.id, gadget2.owner.name);

    // At the end of the method, gadget1 and gadget2 get destroyed, and with
    // them the last counted references to our Owner.  Gadget Man now gets
    // destroyed as well.
}
```

If our requirements change, and we also need to be able to traverse from
Owner->Gadget, we will run into problems: an Rc pointer from Owner->Gadget
introduces a cycle between the objects.  This means that their reference counts
can never reach 0, and the objects will stay alive: a memory leak.  In order to
get around this, we can use `Weak` pointers.  These are reference counted
pointers that don't keep an object alive if there are no normal `Rc` (or
*strong*) pointers left.

Rust actually makes it somewhat difficult to produce this loop in the first
place: in order to end up with two objects that point at each other, one of
them needs to be mutable.  This is problematic because Rc enforces memory
safety by only giving out shared references to the object it wraps, and these
don't allow direct mutation.  We need to wrap the part of the object we wish to
mutate in a `RefCell`, which provides *interior mutability*: a method to
achieve mutability through a shared reference.  `RefCell` enforces Rust's
borrowing rules at runtime.  Read the `Cell` documentation for more details on
interior mutability.

```rust
use std::rc::Rc;
use std::rc::Weak;
use std::cell::RefCell;

struct Owner {
    name: String,
    gadgets: RefCell<Vec<Weak<Gadget>>>
    // ...other fields
}

struct Gadget {
    id: int,
    owner: Rc<Owner>
    // ...other fields
}

fn main() {
    // Create a reference counted Owner.  Note the fact that we've put the
    // Owner's vector of Gadgets inside a RefCell so that we can mutate it
    // through a shared reference.
    let gadget_owner : Rc<Owner> = Rc::new(
            Owner {
                name: "Gadget Man".to_string(),
                gadgets: RefCell::new(Vec::new())
            }
    );

    // Create Gadgets belonging to gadget_owner as before.
    let gadget1 = Rc::new(Gadget{id: 1, owner: gadget_owner.clone()});
    let gadget2 = Rc::new(Gadget{id: 2, owner: gadget_owner.clone()});

    // Add the Gadgets to their Owner.  To do this we mutably borrow from
    // the RefCell holding the Owner's Gadgets.
    gadget_owner.gadgets.borrow_mut().push(gadget1.clone().downgrade());
    gadget_owner.gadgets.borrow_mut().push(gadget2.clone().downgrade());

    // Iterate over our Gadgets, printing their details out
    for gadget_opt in gadget_owner.gadgets.borrow().iter() {

        // gadget_opt is a Weak<Gadget>.  Since weak pointers can't guarantee
        // that their object is still alive, we need to call upgrade() on them
        // to turn them into a strong reference.  This returns an Option, which
        // contains a reference to our object if it still exists.
        let gadget = gadget_opt.upgrade().unwrap();
        println!("Gadget {} owned by {}", gadget.id, gadget.owner.name);
    }

    // At the end of the method, gadget_owner, gadget1 and gadget2 get
    // destroyed.  There are now no strong (Rc) references to the gadgets.
    // Once they get destroyed, the Gadgets get destroyed.  This zeroes the
    // reference count on Gadget Man, so he gets destroyed as well.
}
```

*/

#![stable]

use core::mem::transmute;
use core::cell::Cell;
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::kinds::marker;
use core::ops::{Deref, Drop};
use core::option::{Option, Some, None};
use core::ptr;
use core::ptr::RawPtr;
use core::mem::{min_align_of, size_of};
use core::fmt;

use heap::deallocate;

struct RcBox<T> {
    value: T,
    strong: Cell<uint>,
    weak: Cell<uint>
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
#[stable]
pub struct Rc<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: *mut RcBox<T>,
    _nosend: marker::NoSend,
    _noshare: marker::NoShare
}

#[stable]
impl<T> Rc<T> {
    /// Construct a new reference-counted box
    pub fn new(value: T) -> Rc<T> {
        unsafe {
            Rc {
                // there is an implicit weak pointer owned by all the
                // strong pointers, which ensures that the weak
                // destructor never frees the allocation while the
                // strong destructor is running, even if the weak
                // pointer is stored inside the strong one.
                _ptr: transmute(box RcBox {
                    value: value,
                    strong: Cell::new(1),
                    weak: Cell::new(1)
                }),
                _nosend: marker::NoSend,
                _noshare: marker::NoShare
            }
        }
    }
}

impl<T> Rc<T> {
    /// Downgrade the reference-counted pointer to a weak reference
    #[experimental = "Weak pointers may not belong in this module."]
    pub fn downgrade(&self) -> Weak<T> {
        self.inc_weak();
        Weak {
            _ptr: self._ptr,
            _nosend: marker::NoSend,
            _noshare: marker::NoShare
        }
    }
}

impl<T: Clone> Rc<T> {
    /// Acquires a mutable pointer to the inner contents by guaranteeing that
    /// the reference count is one (no sharing is possible).
    ///
    /// This is also referred to as a copy-on-write operation because the inner
    /// data is cloned if the reference count is greater than one.
    #[inline]
    #[experimental]
    pub fn make_unique(&mut self) -> &mut T {
        // Note that we hold a strong reference, which also counts as
        // a weak reference, so we only clone if there is an
        // additional reference of either kind.
        if self.strong() != 1 || self.weak() != 1 {
            *self = Rc::new(self.deref().clone())
        }
        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the Rc itself to be `mut`, so we're returning the only possible
        // reference to the inner data.
        let inner = unsafe { &mut *self._ptr };
        &mut inner.value
    }
}

#[experimental = "Deref is experimental."]
impl<T> Deref<T> for Rc<T> {
    /// Borrow the value contained in the reference-counted box
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().value
    }
}

#[unsafe_destructor]
#[experimental = "Drop is experimental."]
impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            if !self._ptr.is_null() {
                self.dec_strong();
                if self.strong() == 0 {
                    ptr::read(self.deref()); // destroy the contained object

                    // remove the implicit "strong weak" pointer now
                    // that we've destroyed the contents.
                    self.dec_weak();

                    if self.weak() == 0 {
                        deallocate(self._ptr as *mut u8, size_of::<RcBox<T>>(),
                                   min_align_of::<RcBox<T>>())
                    }
                }
            }
        }
    }
}

#[unstable = "Clone is unstable."]
impl<T> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Rc<T> {
        self.inc_strong();
        Rc { _ptr: self._ptr, _nosend: marker::NoSend, _noshare: marker::NoShare }
    }
}

#[stable]
impl<T: Default> Default for Rc<T> {
    #[inline]
    fn default() -> Rc<T> {
        Rc::new(Default::default())
    }
}

#[unstable = "PartialEq is unstable."]
impl<T: PartialEq> PartialEq for Rc<T> {
    #[inline(always)]
    fn eq(&self, other: &Rc<T>) -> bool { **self == **other }
    #[inline(always)]
    fn ne(&self, other: &Rc<T>) -> bool { **self != **other }
}

#[unstable = "Eq is unstable."]
impl<T: Eq> Eq for Rc<T> {}

#[unstable = "PartialOrd is unstable."]
impl<T: PartialOrd> PartialOrd for Rc<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Rc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    #[inline(always)]
    fn lt(&self, other: &Rc<T>) -> bool { **self < **other }

    #[inline(always)]
    fn le(&self, other: &Rc<T>) -> bool { **self <= **other }

    #[inline(always)]
    fn gt(&self, other: &Rc<T>) -> bool { **self > **other }

    #[inline(always)]
    fn ge(&self, other: &Rc<T>) -> bool { **self >= **other }
}

#[unstable = "Ord is unstable."]
impl<T: Ord> Ord for Rc<T> {
    #[inline]
    fn cmp(&self, other: &Rc<T>) -> Ordering { (**self).cmp(&**other) }
}

#[experimental = "Show is experimental."]
impl<T: fmt::Show> fmt::Show for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// Weak reference to a reference-counted box
#[unsafe_no_drop_flag]
#[experimental = "Weak pointers may not belong in this module."]
pub struct Weak<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: *mut RcBox<T>,
    _nosend: marker::NoSend,
    _noshare: marker::NoShare
}

#[experimental = "Weak pointers may not belong in this module."]
impl<T> Weak<T> {
    /// Upgrade a weak reference to a strong reference
    pub fn upgrade(&self) -> Option<Rc<T>> {
        if self.strong() == 0 {
            None
        } else {
            self.inc_strong();
            Some(Rc { _ptr: self._ptr, _nosend: marker::NoSend, _noshare: marker::NoShare })
        }
    }
}

#[unsafe_destructor]
#[experimental = "Weak pointers may not belong in this module."]
impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        unsafe {
            if !self._ptr.is_null() {
                self.dec_weak();
                // the weak count starts at 1, and will only go to
                // zero if all the strong pointers have disappeared.
                if self.weak() == 0 {
                    deallocate(self._ptr as *mut u8, size_of::<RcBox<T>>(),
                               min_align_of::<RcBox<T>>())
                }
            }
        }
    }
}

#[experimental = "Weak pointers may not belong in this module."]
impl<T> Clone for Weak<T> {
    #[inline]
    fn clone(&self) -> Weak<T> {
        self.inc_weak();
        Weak { _ptr: self._ptr, _nosend: marker::NoSend, _noshare: marker::NoShare }
    }
}

#[doc(hidden)]
trait RcBoxPtr<T> {
    fn inner(&self) -> &RcBox<T>;

    #[inline]
    fn strong(&self) -> uint { self.inner().strong.get() }

    #[inline]
    fn inc_strong(&self) { self.inner().strong.set(self.strong() + 1); }

    #[inline]
    fn dec_strong(&self) { self.inner().strong.set(self.strong() - 1); }

    #[inline]
    fn weak(&self) -> uint { self.inner().weak.get() }

    #[inline]
    fn inc_weak(&self) { self.inner().weak.set(self.weak() + 1); }

    #[inline]
    fn dec_weak(&self) { self.inner().weak.set(self.weak() - 1); }
}

impl<T> RcBoxPtr<T> for Rc<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> { unsafe { &(*self._ptr) } }
}

impl<T> RcBoxPtr<T> for Weak<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> { unsafe { &(*self._ptr) } }
}

#[cfg(test)]
#[allow(experimental)]
mod tests {
    use super::{Rc, Weak};
    use std::cell::RefCell;
    use std::option::{Option, Some, None};
    use std::mem::drop;
    use std::clone::Clone;

    #[test]
    fn test_clone() {
        let x = Rc::new(RefCell::new(5i));
        let y = x.clone();
        *x.borrow_mut() = 20;
        assert_eq!(*y.borrow(), 20);
    }

    #[test]
    fn test_simple() {
        let x = Rc::new(5i);
        assert_eq!(*x, 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Rc::new(5i);
        let y = x.clone();
        assert_eq!(*x, 5);
        assert_eq!(*y, 5);
    }

    #[test]
    fn test_destructor() {
        let x = Rc::new(box 5i);
        assert_eq!(**x, 5);
    }

    #[test]
    fn test_live() {
        let x = Rc::new(5i);
        let y = x.downgrade();
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Rc::new(5i);
        let y = x.downgrade();
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn gc_inside() {
        // see issue #11532
        use std::gc::GC;
        let a = Rc::new(RefCell::new(box(GC) 1i));
        assert!(a.try_borrow_mut().is_some());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: RefCell<Option<Weak<Cycle>>>
        }

        let a = Rc::new(Cycle { x: RefCell::new(None) });
        let b = a.clone().downgrade();
        *a.x.borrow_mut() = Some(b);

        // hopefully we don't double-free (or leak)...
    }

    #[test]
    fn test_cowrc_clone_make_unique() {
        let mut cow0 = Rc::new(75u);
        let mut cow1 = cow0.clone();
        let mut cow2 = cow1.clone();

        assert!(75 == *cow0.make_unique());
        assert!(75 == *cow1.make_unique());
        assert!(75 == *cow2.make_unique());

        *cow0.make_unique() += 1;
        *cow1.make_unique() += 2;
        *cow2.make_unique() += 3;

        assert!(76 == *cow0);
        assert!(77 == *cow1);
        assert!(78 == *cow2);

        // none should point to the same backing memory
        assert!(*cow0 != *cow1);
        assert!(*cow0 != *cow2);
        assert!(*cow1 != *cow2);
    }

    #[test]
    fn test_cowrc_clone_unique2() {
        let mut cow0 = Rc::new(75u);
        let cow1 = cow0.clone();
        let cow2 = cow1.clone();

        assert!(75 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        *cow0.make_unique() += 1;

        assert!(76 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        // cow1 and cow2 should share the same contents
        // cow0 should have a unique reference
        assert!(*cow0 != *cow1);
        assert!(*cow0 != *cow2);
        assert!(*cow1 == *cow2);
    }

    #[test]
    fn test_cowrc_clone_weak() {
        let mut cow0 = Rc::new(75u);
        let cow1_weak = cow0.downgrade();

        assert!(75 == *cow0);
        assert!(75 == *cow1_weak.upgrade().unwrap());

        *cow0.make_unique() += 1;

        assert!(76 == *cow0);
        assert!(cow1_weak.upgrade().is_none());
    }

}
