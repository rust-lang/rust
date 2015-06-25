% Working With Uninitialized Memory

All runtime-allocated memory in a Rust program begins its life as
*uninitialized*. In this state the value of the memory is an indeterminate pile
of bits that may or may not even reflect a valid state for the type that is
supposed to inhabit that location of memory. Attempting to interpret this memory
as a value of *any* type will cause Undefined Behaviour. Do Not Do This.

Like C, all stack variables in Rust begin their life as uninitialized until a
value is explicitly assigned to them. Unlike C, Rust statically prevents you
from ever reading them until you do:

```rust
fn main() {
	let x: i32;
	println!("{}", x);
}
```

```text
src/main.rs:3:20: 3:21 error: use of possibly uninitialized variable: `x`
src/main.rs:3     println!("{}", x);
                                 ^
```

This is based off of a basic branch analysis: every branch must assign a value
to `x` before it is first used. Interestingly, Rust doesn't require the variable
to be mutable to perform a delayed initialization if every branch assigns
exactly once. However the analysis does not take advantage of constant analysis
or anything like that. So this compiles:

```rust
fn main() {
	let x: i32;
	let y: i32;

	y = 1;

	if true {
		x = 1;
	} else {
		x = 2;
	}

    println!("{} {}", x, y);
}
```

but this doesn't:

```rust
fn main() {
	let x: i32;
	if true {
		x = 1;
	}
	println!("{}", x);
}
```

```text
src/main.rs:6:17: 6:18 error: use of possibly uninitialized variable: `x`
src/main.rs:6 	println!("{}", x);
```

while this does:

```rust
fn main() {
	let x: i32;
	if true {
		x = 1;
		println!("{}", x);
	}
	// Don't care that there are branches where it's not initialized
	// since we don't use the value in those branches
}
```

If a value is moved out of a variable, that variable becomes logically
uninitialized if the type of the value isn't Copy. That is:

```rust
fn main() {
	let x = 0;
	let y = Box::new(0);
	let z1 = x; // x is still valid because i32 is Copy
	let z2 = y; // y is now logically uninitialized because Box isn't Copy
}
```

However reassigning `y` in this example *would* require `y` to be marked as
mutable, as a Safe Rust program could observe that the value of `y` changed.
Otherwise the variable is exactly like new.

This raises an interesting question with respect to `Drop`: where does Rust try
to call the destructor of a variable that is conditionally initialized? It turns
out that Rust actually tracks whether a type should be dropped or not *at
runtime*. As a variable becomes initialized and uninitialized, a *drop flag* for
that variable is set and unset. When a variable goes out of scope or is assigned
it evaluates whether the current value of the variable should be dropped. Of
course, static analysis can remove these checks. If the compiler can prove that
a value is guaranteed to be either initialized or not, then it can theoretically
generate more efficient code! As such it may be desirable to structure code to
have *static drop semantics* when possible.

As of Rust 1.0, the drop flags are actually not-so-secretly stashed in a secret
field of any type that implements Drop. The language sets the drop flag by
overwriting the entire struct with a particular value. This is pretty obviously
Not The Fastest and causes a bunch of trouble with optimizing code. As such work
is currently under way to move the flags out onto the stack frame where they
more reasonably belong. Unfortunately this work will take some time as it
requires fairly substantial changes to the compiler.

So in general, Rust programs don't need to worry about uninitialized values on
the stack for correctness. Although they might care for performance. Thankfully,
Rust makes it easy to take control here! Uninitialized values are there, and
Safe Rust lets you work with them, but you're never in trouble.

One interesting exception to this rule is working with arrays. Safe Rust doesn't
permit you to partially initialize an array. When you initialize an array, you
can either set every value to the same thing with `let x = [val; N]`, or you can
specify each member individually with `let x = [val1, val2, val3]`.
Unfortunately this is pretty rigid, especially if you need to initialize your
array in a more incremental or dynamic way.

Unsafe Rust gives us a powerful tool to handle this problem:
`std::mem::uninitialized`. This function pretends to return a value when really
it does nothing at all. Using it, we can convince Rust that we have initialized
a variable, allowing us to do trickier things with conditional and incremental
initialization.

Unfortunately, this raises a tricky problem. Assignment has a different meaning
to Rust based on whether it believes that a variable is initialized or not. If
it's uninitialized, then Rust will semantically just memcopy the bits over the
uninit ones, and do nothing else. However if Rust believes a value to be
initialized, it will try to `Drop` the old value! Since we've tricked Rust into
believing that the value is initialized, we can no longer safely use normal
assignment.

This is also a problem if you're working with a raw system allocator, which of
course returns a pointer to uninitialized memory.

To handle this, we must use the `std::ptr` module. In particular, it provides
three functions that allow us to assign bytes to a location in memory without
evaluating the old value: `write`, `copy`, and `copy_nonoverlapping`.

* `ptr::write(ptr, val)` takes a `val` and moves it into the address pointed
  to by `ptr`.
* `ptr::copy(src, dest, count)` copies the bits that `count` T's would occupy
  from src to dest. (this is equivalent to memmove -- note that the argument
  order is reversed!)
* `ptr::copy_nonoverlapping(src, dest, count)` does what `copy` does, but a
  little faster on the assumption that the two ranges of memory don't overlap.
  (this is equivalent to memcopy -- note that the argument order is reversed!)

It should go without saying that these functions, if misused, will cause serious
havoc or just straight up Undefined Behaviour. The only things that these
functions *themselves* require is that the locations you want to read and write
are allocated. However the ways writing arbitrary bit patterns to arbitrary
locations of memory can break things are basically uncountable!

Putting this all together, we get the following:

```rust
fn main() {
	use std::mem;

	// size of the array is hard-coded but easy to change. This means we can't
	// use [a, b, c] syntax to initialize the array, though!
	const SIZE = 10;

	let x: [Box<u32>; SIZE];

	unsafe {
		// convince Rust that x is Totally Initialized
		x = mem::uninitialized();
		for i in 0..SIZE {
			// very carefully overwrite each index without reading it
			ptr::write(&mut x[i], Box::new(i));
		}
	}

	println!("{}", x);
}
```

It's worth noting that you don't need to worry about ptr::write-style
shenanigans with Plain Old Data (POD; types which don't implement Drop, nor
contain Drop types), because Rust knows not to try to Drop them. Similarly you
should be able to assign the POD fields of partially initialized structs
directly.

However when working with uninitialized memory you need to be ever vigilant for
Rust trying to Drop values you make like this before they're fully initialized.
So every control path through that variable's scope must initialize the value
before it ends. *This includes code panicking*. Again, POD types need not worry.

And that's about it for working with uninitialized memory! Basically nothing
anywhere expects to be handed uninitialized memory, so if you're going to pass
it around at all, be sure to be *really* careful.
