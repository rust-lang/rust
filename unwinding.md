% Unwinding

Rust has a *tiered* error-handling scheme:

* If something might reasonably be absent, Option is used
* If something goes wrong and can reasonably be handled, Result is used
* If something goes wrong and cannot reasonably be handled, the thread panics
* If something catastrophic happens, the program aborts

Option and Result are overwhelmingly preferred in most situations, especially
since they can be promoted into a panic or abort at the API user's discretion.
However, anything and everything *can* panic, and you need to be ready for this.
Panics cause the thread to halt normal execution and unwind its stack, calling
destructors as if every function instantly returned.

As of 1.0, Rust is of two minds when it comes to panics. In the long-long-ago,
Rust was much more like Erlang. Like Erlang, Rust had lightweight tasks,
and tasks were intended to kill themselves with a panic when they reached an
untenable state. Unlike an exception in Java or C++, a panic could not be
caught at any time. Panics could only be caught by the owner of the task, at which
point they had to be handled or *that* task would itself panic.

Unwinding was important to this story because if a task's
destructors weren't called, it would cause memory and other system resources to
leak. Since tasks were expected to die during normal execution, this would make
Rust very poor for long-running systems!

As the Rust we know today came to be, this style of programming grew out of
fashion in the push for less-and-less abstraction. Light-weight tasks were
killed in the name of heavy-weight OS threads. Still, panics could only be
caught by the parent thread. This meant catching a panic required spinning up
an entire OS thread! Although Rust maintains the philosophy that panics should
not be used for "basic" error-handling like C++ or Java, it is still desirable
to not have the entire program crash in the face of a panic.

In the near future there will be a stable interface for catching panics in an
arbitrary location, though we would encourage you to still only do this
sparingly. In particular, Rust's current unwinding implementation is heavily
optimized for the "doesn't unwind" case. If a program doesn't unwind, there
should be no runtime cost for the program being *ready* to unwind. As a
consequence, *actually* unwinding will be more expensive than in e.g. Java.
Don't build your programs to unwind under normal circumstances. Ideally, you
should only panic for programming errors.




# Exception Safety

Being ready for unwinding is often referred to as "exception safety"
in the broader programming world. In Rust, their are two levels of exception
safety that one may concern themselves with:

* In unsafe code, we *must* be exception safe to the point of not violating
  memory safety.

* In safe code, it is *good* to be exception safe to the point of your program
  doing the right thing.

As is the case in many places in Rust, unsafe code must be ready to deal with
bad safe code, and that includes code that panics. Code that transiently creates
unsound states must be careful that a panic does not cause that state to be
used. Generally this means ensuring that only non-panicing code is run while
these states exist, or making a guard that cleans up the state in the case of
a panic. This does not necessarily mean that the state a panic witnesses is a
fully *coherent* state. We need only guarantee that it's a *safe* state.

For instance, consider extending a Vec:

```rust

impl Extend<T> for Vec<T> {
	fn extend<I: IntoIter<Item=T>>(&mut self, iterable: I) {
		let mut iter = iterable.into_iter();
		let size_hint = iter.size_hint().0;
		self.reserve(size_hint);
		self.set_len(self.len() + size_hint());

		for
	}
}

