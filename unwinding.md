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
caught by the parent thread. This means catching a panic requires spinning up
an entire OS thread! This unfortunately stands in conflict to Rust's philosophy
of zero-cost abstractions.

In the near future there will be a stable interface for catching panics in an
arbitrary location, though we would encourage you to still only do this
sparingly. In particular, Rust's current unwinding implementation is heavily
optimized for the "doesn't unwind" case. If a program doesn't unwind, there
should be no runtime cost for the program being *ready* to unwind. As a
consequence, *actually* unwinding will be more expensive than in e.g. Java.
Don't build your programs to unwind under normal circumstances. Ideally, you
should only panic for programming errors or *extreme* problems.




# Exception Safety

Being ready for unwinding is often referred to as *exception safety*
in the broader programming world. In Rust, their are two levels of exception
safety that one may concern themselves with:

* In unsafe code, we *must* be exception safe to the point of not violating
  memory safety.

* In safe code, it is *good* to be exception safe to the point of your program
  doing the right thing.

As is the case in many places in Rust, unsafe code must be ready to deal with
bad safe code, and that includes code that panics. Code that transiently creates
unsound states must be careful that a panic does not cause that state to be
used. Generally this means ensuring that only non-panicking code is run while
these states exist, or making a guard that cleans up the state in the case of
a panic. This does not necessarily mean that the state a panic witnesses is a
fully *coherent* state. We need only guarantee that it's a *safe* state.

Most unsafe code is leaf-like, and therefore fairly easy to make exception-safe.
It controls all the code that runs, and most of that code can't panic. However
it is often the case that code that works with arrays works with temporarily
uninitialized data while repeatedly invoking caller-provided code. Such code
needs to be careful, and consider exception-safety.





## Vec::push_all

`Vec::push_all` is a temporary hack to get extending a Vec by a slice reliably
effecient without specialization. Here's a simple implementation:

```rust,ignore
impl<T: Clone> Vec<T> {
    fn push_all(&mut self, to_push: &[T]) {
        self.reserve(to_push.len());
        unsafe {
            // can't overflow because we just reserved this
            self.set_len(self.len() + to_push.len());

            for (i, x) in to_push.iter().enumerate() {
                self.ptr().offset(i as isize).write(x.clone());
            }
        }
    }
}
```

We bypass `push` in order to avoid redundant capacity and `len` checks on the
Vec that we definitely know has capacity. The logic is totally correct, except
there's a subtle problem with our code: it's not exception-safe! `set_len`,
`offset`, and `write` are all fine, but *clone* is the panic bomb we over-looked.

Clone is completely out of our control, and is totally free to panic. If it does,
our function will exit early with the length of the Vec set too large. If
the Vec is looked at or dropped, uninitialized memory will be read!

The fix in this case is fairly simple. If we want to guarantee that the values
we *did* clone are dropped we can set the len *in* the loop. If we just want to
guarantee that uninitialized memory can't be observed, we can set the len *after*
the loop.





## BinaryHeap::sift_up

Bubbling an element up a heap is a bit more complicated than extending a Vec.
The pseudocode is as follows:

```text
bubble_up(heap, index):
    while index != 0 && heap[index] < heap[parent(index)]:
        heap.swap(index, parent(index))
        index = parent(index)

```

A literal transcription of this code to Rust is totally fine, but has an annoying
performance characteristic: the `self` element is swapped over and over again
uselessly. We would *rather* have the following:

```text
bubble_up(heap, index):
    let elem = heap[index]
    while index != 0 && element < heap[parent(index)]:
        heap[index] = heap[parent(index)]
        index = parent(index)
    heap[index] = elem
```

This code ensures that each element is copied as little as possible (it is in
fact necessary that elem be copied twice in general). However it now exposes
some exception-safety trouble! At all times, there exists two copies of one
value. If we panic in this function something will be double-dropped.
Unfortunately, we also don't have full control of the code: that comparison is
user-defined!

Unlike Vec, the fix isn't as easy here. One option is to break the user-defined
code and the unsafe code into two separate phases:

```text
bubble_up(heap, index):
    let end_index = index;
    while end_index != 0 && heap[end_index] < heap[parent(end_index)]:
        end_index = parent(end_index)

    let elem = heap[index]
    while index != end_index:
        heap[index] = heap[parent(index)]
        index = parent(index)
    heap[index] = elem
```

If the user-defined code blows up, that's no problem anymore, because we haven't
actually touched the state of the heap yet. Once we do start messing with the
heap, we're working with only data and functions that we trust, so there's no
concern of panics.

Perhaps you're not happy with this design. Surely, it's cheating! And we have
to do the complex heap traversal *twice*! Alright, let's bite the bullet. Let's
intermix untrusted and unsafe code *for reals*.

If Rust had `try` and `finally` like in Java, we could do the following:

```text
bubble_up(heap, index):
    let elem = heap[index]
    try:
        while index != 0 && element < heap[parent(index)]:
            heap[index] = heap[parent(index)]
            index = parent(index)
    finally:
        heap[index] = elem
```

The basic idea is simple: if the comparison panics, we just toss the loose
element in the logically uninitialized index and bail out. Anyone who observes
the heap will see a potentially *inconsistent* heap, but at least it won't
cause any double-drops! If the algorithm terminates normally, then this
operation happens to coincide precisely with the how we finish up regardless.

Sadly, Rust has no such construct, so we're going to need to roll our own! The
way to do this is to store the algorithm's state in a separate struct with a
destructor for the "finally" logic. Whether we panic or not, that destructor
will run and clean up after us.

```rust
struct Hole<'a, T: 'a> {
    data: &'a mut [T],
    /// `elt` is always `Some` from new until drop.
    elt: Option<T>,
    pos: usize,
}

impl<'a, T> Hole<'a, T> {
    fn new(data: &'a mut [T], pos: usize) -> Self {
        unsafe {
            let elt = ptr::read(&data[pos]);
            Hole {
                data: data,
                elt: Some(elt),
                pos: pos,
            }
        }
    }

    fn pos(&self) -> usize { self.pos }

    fn removed(&self) -> &T { self.elt.as_ref().unwrap() }

    unsafe fn get(&self, index: usize) -> &T { &self.data[index] }

    unsafe fn move_to(&mut self, index: usize) {
        let index_ptr: *const _ = &self.data[index];
        let hole_ptr = &mut self.data[self.pos];
        ptr::copy_nonoverlapping(index_ptr, hole_ptr, 1);
        self.pos = index;
    }
}

impl<'a, T> Drop for Hole<'a, T> {
    fn drop(&mut self) {
        // fill the hole again
        unsafe {
            let pos = self.pos;
            ptr::write(&mut self.data[pos], self.elt.take().unwrap());
        }
    }
}

impl<T: Ord> BinaryHeap<T> {
    fn sift_up(&mut self, pos: usize) {
        unsafe {
            // Take out the value at `pos` and create a hole.
            let mut hole = Hole::new(&mut self.data, pos);

            while hole.pos() != 0 {
                let parent = parent(hole.pos());
                if hole.removed() <= hole.get(parent) { break }
                hole.move_to(parent);
            }
            // Hole will be unconditionally filled here; panic or not!
        }
    }
}
```




## Poisoning

Although all unsafe code *must* ensure some minimal level of exception safety,
some types may choose to explicitly *poison* themselves if they witness a panic.
Poisoning doesn't entail anything in particular. Generally it just means
preventing normal usage from proceeding. The most notable example of this is the
standard library's Mutex type. A Mutex will poison itself if one of its
MutexGuards (the thing it returns when a lock is obtained) is dropped during a
panic. Any future attempts to lock the Mutex will return an `Err`.

Mutex poisons not for *true* safety in the sense that Rust normally cares about. It
poisons as a safety-guard against blindly using the data that comes out of a Mutex
that has witnessed a panic while locked. The data in such a Mutex was likely in the
middle of being modified, and as such may be in an inconsistent or incomplete state.
It is important to note that one cannot violate memory safety with such a type
if it is correctly written. After all, it must be minimally exception safe!

However if the Mutex contained, say, a BinaryHeap that does not actually have the
heap property, it's unlikely that any code that uses it will do
what the author intended. As such, the program should not proceed normally.
Still, if you're double-plus-sure that you can do *something* with the value,
the Err exposes a method to get the lock anyway. It *is* safe, after all.



# FFI

Rust's unwinding strategy is not specified to be fundamentally compatible
with any other language's unwinding. As such, unwinding into Rust from another
language, or unwinding into another language from Rust is Undefined Behaviour.
You must *absolutely* catch any panics at the FFI boundary! What you do at that
point is up to you, but *something* must be done. If you fail to do this,
at best, your application will crash and burn. At worst, your application *won't*
crash and burn, and will proceed with completely clobbered state.
