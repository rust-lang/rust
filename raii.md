% The Perils Of RAII

Ownership Based Resource Management (AKA RAII: Resource Acquisition is Initialization) is
something you'll interact with a lot in Rust. Especially if you use the standard library.

Roughly speaking the pattern is as follows: to acquire a resource, you create an object that
manages it. To release the resource, you simply destroy the object, and it cleans up the
resource for you. The most common "resource"
this pattern manages is simply *memory*. `Box`, `Rc`, and basically everything in
`std::collections` is a convenience to enable correctly managing memory. This is particularly
important in Rust because we have no pervasive GC to rely on for memory management. Which is the
point, really: Rust is about control. However we are not limited to just memory.
Pretty much every other system resource like a thread, file, or socket is exposed through
this kind of API.

So, how does RAII work in Rust? Unlike C++, Rust does not come with a slew on builtin
kinds of constructor. There are no Copy, Default, Assignment, Move, or whatever constructors.
This largely has to do with Rust's philosophy of being explicit.

Move constructors are meaningless in Rust because we don't enable types to "care" about their
location in memory. Every type must be ready for it to be blindly memcopied to somewhere else
in memory. This means pure on-the-stack-but-still-movable intrusive linked lists are simply
not happening in Rust (safely).

Assignment and copy constructors similarly don't exist because move semantics are the *default*
in rust. At most `x = y` just moves the bits of y into the x variable. Rust does provide two
facilities for going back to C++'s copy-oriented semantics: `Copy` and `Clone`. Clone is our
moral equivalent of copy constructor, but it's never implicitly invoked. You have to explicitly
call `clone` on an element you want to be cloned. Copy is a special case of Clone where the
implementation is just "duplicate the bitwise representation". Copy types *are* implicitely
cloned whenever they're moved, but because of the definition of Copy this just means *not*
treating the old copy as uninitialized; a no-op.

While Rust provides a `Default` trait for specifying the moral equivalent of a default
constructor, it's incredibly rare for this trait to be used. This is because variables
aren't implicitely initialized (see [working with uninitialized memory][uninit] for details).
Default is basically only useful for generic programming.

More often than not, in a concrete case a type will provide a static `new` method for any
kind of "default" constructor. This has no relation to `new` in other languages and has no
special meaning. It's just a naming convention.

What the language *does* provide is full-blown automatic destructors through the `Drop` trait,
which provides the following method:

```rust
fn drop(&mut self);
```

This method gives the type time to somehow finish what it was doing. **After `drop` is run,
Rust will recursively try to drop all of the fields of the `self` struct**. This is a
convenience feature so that you don't have to write "destructor boilerplate" dropping
children. **There is no way to prevent this in Rust 1.0**.  Also note that `&mut self` means
that even if you *could* supress recursive Drop, Rust will prevent you from e.g. moving fields
out of self. For most types, this is totally fine: they own all their data, there's no
additional state passed into drop to try to send it to, and `self` is about to be marked as
uninitialized (and therefore inaccessible).

For instance, a custom implementation of `Box` might write `Drop` like this:

```rust
struct Box<T>{ ptr: *mut T }

impl<T> Drop for Box<T> {
	fn drop(&mut self) {
		unsafe {
			(*self.ptr).drop();
			heap::deallocate(self.ptr);
		}
	}
}
```

and this works fine because when Rust goes to drop the `ptr` field it just sees a *mut that
has no actual `Drop` implementation. Similarly nothing can use-after-free the `ptr` because
the Box is completely gone.

However this wouldn't work:

```rust
struct Box<T>{ ptr: *mut T }

impl<T> Drop for Box<T> {
	fn drop(&mut self) {
		unsafe {
			(*self.ptr).drop();
			heap::deallocate(self.ptr);
		}
	}
}

struct SuperBox<T> { box: Box<T> }

impl<T> Drop for SuperBox<T> {
	fn drop(&mut self) {
		unsafe {
			// Hyper-optimized: deallocate the box's contents for it
			// without `drop`ing the contents
			heap::deallocate(self.box.ptr);
		}
	}
}
```

because after we deallocate the `box`'s ptr in SuperBox's destructor, Rust will
happily proceed to tell the box to Drop itself and everything will blow up with
use-after-frees and double-frees.

Note that the recursive drop behaviour applies to *all* structs and enums
regardless of whether they implement Drop. Therefore something like

```rust
struct Boxy<T> {
	data1: Box<T>,
	data2: Box<T>,
	info: u32,
}
```

will have its data1 and data2's fields destructors whenever it "would" be
dropped, even though it itself doesn't implement Drop. We say that such a type
*needs Drop*, even though it is not itself Drop.

Similarly,

```rust
enum Link {
	Next(Box<Link>),
	None,
}
```

will have its inner Box field dropped *if and only if* a value stores the Next variant.

In general this works really nice because you don't need to worry about adding/removing
dtors when you refactor your data layout. Still there's certainly many valid usecases for
needing to do trickier things with destructors.

The classic safe solution to blocking recursive drop semantics and allowing moving out
of Self is to use an Option:

```rust
struct Box<T>{ ptr: *mut T }

impl<T> Drop for Box<T> {
	fn drop(&mut self) {
		unsafe {
			(*self.ptr).drop();
			heap::deallocate(self.ptr);
		}
	}
}

struct SuperBox<T> { box: Option<Box<T>> }

impl<T> Drop for SuperBox<T> {
	fn drop(&mut self) {
		unsafe {
			// Hyper-optimized: deallocate the box's contents for it
			// without `drop`ing the contents. Need to set the `box`
			// fields as `None` to prevent Rust from trying to Drop it.
			heap::deallocate(self.box.take().unwrap().ptr);
		}
	}
}
```

However this has fairly odd semantics: you're saying that a field that *should* always be Some
may be None, just because that happens in the dtor. Of course this conversely makes a lot of sense:
you can call arbitrary methods on self during the destructor, and this should prevent you from
ever doing so after deinitializing the field. Not that it will prevent you from producing any other
arbitrarily invalid state in there.

On balance this is an ok choice. Certainly if you're just getting started.

In the future, we expect there to be a first-class way to announce that a field
should be automatically dropped.

[uninit]: uninitialized.html