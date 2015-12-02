% Unchecked Uninitialized Memory

One interesting exception to this rule is working with arrays. Safe Rust doesn't
permit you to partially initialize an array. When you initialize an array, you
can either set every value to the same thing with `let x = [val; N]`, or you can
specify each member individually with `let x = [val1, val2, val3]`.
Unfortunately this is pretty rigid, especially if you need to initialize your
array in a more incremental or dynamic way.

Unsafe Rust gives us a powerful tool to handle this problem:
`mem::uninitialized`. This function pretends to return a value when really
it does nothing at all. Using it, we can convince Rust that we have initialized
a variable, allowing us to do trickier things with conditional and incremental
initialization.

Unfortunately, this opens us up to all kinds of problems. Assignment has a
different meaning to Rust based on whether it believes that a variable is
initialized or not. If it's believed uninitialized, then Rust will semantically
just memcopy the bits over the uninitialized ones, and do nothing else. However
if Rust believes a value to be initialized, it will try to `Drop` the old value!
Since we've tricked Rust into believing that the value is initialized, we can no
longer safely use normal assignment.

This is also a problem if you're working with a raw system allocator, which
returns a pointer to uninitialized memory.

To handle this, we must use the `ptr` module. In particular, it provides
three functions that allow us to assign bytes to a location in memory without
dropping the old value: `write`, `copy`, and `copy_nonoverlapping`.

* `ptr::write(ptr, val)` takes a `val` and moves it into the address pointed
  to by `ptr`.
* `ptr::copy(src, dest, count)` copies the bits that `count` T's would occupy
  from src to dest. (this is equivalent to memmove -- note that the argument
  order is reversed!)
* `ptr::copy_nonoverlapping(src, dest, count)` does what `copy` does, but a
  little faster on the assumption that the two ranges of memory don't overlap.
  (this is equivalent to memcpy -- note that the argument order is reversed!)

It should go without saying that these functions, if misused, will cause serious
havoc or just straight up Undefined Behavior. The only things that these
functions *themselves* require is that the locations you want to read and write
are allocated. However the ways writing arbitrary bits to arbitrary
locations of memory can break things are basically uncountable!

Putting this all together, we get the following:

```rust
use std::mem;
use std::ptr;

// Size of the array is hard-coded but easy to change. This means we can't
// use [a, b, c] syntax to initialize the array, though!
const SIZE: usize = 10;

let mut x: [Box<u32>; SIZE];

unsafe {
	// Convince Rust that x is Totally Initialized.
	x = mem::uninitialized();
	for i in 0..SIZE {
		// Very carefully overwrite each index without reading it.
		// NOTE: exception safety is not a concern; Box can't panic.
		ptr::write(&mut x[i], Box::new(i as u32));
	}
}

println!("{:?}", x);
```

It's worth noting that you don't need to worry about `ptr::write`-style
shenanigans with types which don't implement `Drop` or contain `Drop` types,
because Rust knows not to try to drop them. Similarly you should be able to
assign to fields of partially initialized structs directly if those fields don't
contain any `Drop` types.

However when working with uninitialized memory you need to be ever-vigilant for
Rust trying to drop values you make like this before they're fully initialized.
Every control path through that variable's scope must initialize the value
before it ends, if it has a destructor.
*[This includes code panicking](unwinding.html)*.

And that's about it for working with uninitialized memory! Basically nothing
anywhere expects to be handed uninitialized memory, so if you're going to pass
it around at all, be sure to be *really* careful.
