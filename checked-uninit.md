% Checked Uninitialized Memory

Like C, all stack variables in Rust are uninitialized until a
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

	if true {
		x = 1;
	} else {
		x = 2;
	}

    println!("{}", x);
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
a value, it evaluates whether the current value of the variable should be dropped.
Of course, static analysis can remove these checks. If the compiler can prove that
a value is guaranteed to be either initialized or not, then it can theoretically
generate more efficient code! As such it may be desirable to structure code to
have *static drop semantics* when possible.

As of Rust 1.0, the drop flags are actually not-so-secretly stashed in a hidden
field of any type that implements Drop. The language sets the drop flag by
overwriting the entire struct with a particular value. This is pretty obviously
Not The Fastest and causes a bunch of trouble with optimizing code. As such work
is currently under way to move the flags out onto the stack frame where they
more reasonably belong. Unfortunately this work will take some time as it
requires fairly substantial changes to the compiler.

So in general, Rust programs don't need to worry about uninitialized values on
the stack for correctness. Although they might care for performance. Thankfully,
Rust makes it easy to take control here! Uninitialized values are there, and
Safe Rust lets you work with them, but you're never in danger.
