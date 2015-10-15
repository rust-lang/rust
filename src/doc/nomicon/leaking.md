% Leaking

Ownership-based resource management is intended to simplify composition. You
acquire resources when you create the object, and you release the resources when
it gets destroyed. Since destruction is handled for you, it means you can't
forget to release the resources, and it happens as soon as possible! Surely this
is perfect and all of our problems are solved.

Everything is terrible and we have new and exotic problems to try to solve.

Many people like to believe that Rust eliminates resource leaks. In practice,
this is basically true. You would be surprised to see a Safe Rust program
leak resources in an uncontrolled way.

However from a theoretical perspective this is absolutely not the case, no
matter how you look at it. In the strictest sense, "leaking" is so abstract as
to be unpreventable. It's quite trivial to initialize a collection at the start
of a program, fill it with tons of objects with destructors, and then enter an
infinite event loop that never refers to it. The collection will sit around
uselessly, holding on to its precious resources until the program terminates (at
which point all those resources would have been reclaimed by the OS anyway).

We may consider a more restricted form of leak: failing to drop a value that is
unreachable. Rust also doesn't prevent this. In fact Rust *has a function for
doing this*: `mem::forget`. This function consumes the value it is passed *and
then doesn't run its destructor*.

In the past `mem::forget` was marked as unsafe as a sort of lint against using
it, since failing to call a destructor is generally not a well-behaved thing to
do (though useful for some special unsafe code). However this was generally
determined to be an untenable stance to take: there are many ways to fail to
call a destructor in safe code. The most famous example is creating a cycle of
reference-counted pointers using interior mutability.

It is reasonable for safe code to assume that destructor leaks do not happen, as
any program that leaks destructors is probably wrong. However *unsafe* code
cannot rely on destructors to be run in order to be safe. For most types this
doesn't matter: if you leak the destructor then the type is by definition
inaccessible, so it doesn't matter, right? For instance, if you leak a `Box<u8>`
then you waste some memory but that's hardly going to violate memory-safety.

However where we must be careful with destructor leaks are *proxy* types. These
are types which manage access to a distinct object, but don't actually own it.
Proxy objects are quite rare. Proxy objects you'll need to care about are even
rarer. However we'll focus on three interesting examples in the standard
library:

* `vec::Drain`
* `Rc`
* `thread::scoped::JoinGuard`



## Drain

`drain` is a collections API that moves data out of the container without
consuming the container. This enables us to reuse the allocation of a `Vec`
after claiming ownership over all of its contents. It produces an iterator
(Drain) that returns the contents of the Vec by-value.

Now, consider Drain in the middle of iteration: some values have been moved out,
and others haven't. This means that part of the Vec is now full of logically
uninitialized data! We could backshift all the elements in the Vec every time we
remove a value, but this would have pretty catastrophic performance
consequences.

Instead, we would like Drain to fix the Vec's backing storage when it is
dropped. It should run itself to completion, backshift any elements that weren't
removed (drain supports subranges), and then fix Vec's `len`. It's even
unwinding-safe! Easy!

Now consider the following:

```rust,ignore
let mut vec = vec![Box::new(0); 4];

{
    // start draining, vec can no longer be accessed
    let mut drainer = vec.drain(..);

    // pull out two elements and immediately drop them
    drainer.next();
    drainer.next();

    // get rid of drainer, but don't call its destructor
    mem::forget(drainer);
}

// Oops, vec[0] was dropped, we're reading a pointer into free'd memory!
println!("{}", vec[0]);
```

This is pretty clearly Not Good. Unfortunately, we're kind of stuck between a
rock and a hard place: maintaining consistent state at every step has an
enormous cost (and would negate any benefits of the API). Failing to maintain
consistent state gives us Undefined Behavior in safe code (making the API
unsound).

So what can we do? Well, we can pick a trivially consistent state: set the Vec's
len to be 0 when we start the iteration, and fix it up if necessary in the
destructor. That way, if everything executes like normal we get the desired
behavior with minimal overhead. But if someone has the *audacity* to
mem::forget us in the middle of the iteration, all that does is *leak even more*
(and possibly leave the Vec in an unexpected but otherwise consistent state).
Since we've accepted that mem::forget is safe, this is definitely safe. We call
leaks causing more leaks a *leak amplification*.




## Rc

Rc is an interesting case because at first glance it doesn't appear to be a
proxy value at all. After all, it manages the data it points to, and dropping
all the Rcs for a value will drop that value. Leaking an Rc doesn't seem like it
would be particularly dangerous. It will leave the refcount permanently
incremented and prevent the data from being freed or dropped, but that seems
just like Box, right?

Nope.

Let's consider a simplified implementation of Rc:

```rust,ignore
struct Rc<T> {
    ptr: *mut RcBox<T>,
}

struct RcBox<T> {
    data: T,
    ref_count: usize,
}

impl<T> Rc<T> {
    fn new(data: T) -> Self {
        unsafe {
            // Wouldn't it be nice if heap::allocate worked like this?
            let ptr = heap::allocate<RcBox<T>>();
            ptr::write(ptr, RcBox {
                data: data,
                ref_count: 1,
            });
            Rc { ptr: ptr }
        }
    }

    fn clone(&self) -> Self {
        unsafe {
            (*self.ptr).ref_count += 1;
        }
        Rc { ptr: self.ptr }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            (*self.ptr).ref_count -= 1;
            if (*self.ptr).ref_count == 0 {
                // drop the data and then free it
                ptr::read(self.ptr);
                heap::deallocate(self.ptr);
            }
        }
    }
}
```

This code contains an implicit and subtle assumption: `ref_count` can fit in a
`usize`, because there can't be more than `usize::MAX` Rcs in memory. However
this itself assumes that the `ref_count` accurately reflects the number of Rcs
in memory, which we know is false with `mem::forget`. Using `mem::forget` we can
overflow the `ref_count`, and then get it down to 0 with outstanding Rcs. Then
we can happily use-after-free the inner data. Bad Bad Not Good.

This can be solved by just checking the `ref_count` and doing *something*. The
standard library's stance is to just abort, because your program has become
horribly degenerate. Also *oh my gosh* it's such a ridiculous corner case.




## thread::scoped::JoinGuard

The thread::scoped API intends to allow threads to be spawned that reference
data on their parent's stack without any synchronization over that data by
ensuring the parent joins the thread before any of the shared data goes out
of scope.

```rust,ignore
pub fn scoped<'a, F>(f: F) -> JoinGuard<'a>
    where F: FnOnce() + Send + 'a
```

Here `f` is some closure for the other thread to execute. Saying that
`F: Send +'a` is saying that it closes over data that lives for `'a`, and it
either owns that data or the data was Sync (implying `&data` is Send).

Because JoinGuard has a lifetime, it keeps all the data it closes over
borrowed in the parent thread. This means the JoinGuard can't outlive
the data that the other thread is working on. When the JoinGuard *does* get
dropped it blocks the parent thread, ensuring the child terminates before any
of the closed-over data goes out of scope in the parent.

Usage looked like:

```rust,ignore
let mut data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
{
    let guards = vec![];
    for x in &mut data {
        // Move the mutable reference into the closure, and execute
        // it on a different thread. The closure has a lifetime bound
        // by the lifetime of the mutable reference `x` we store in it.
        // The guard that is returned is in turn assigned the lifetime
        // of the closure, so it also mutably borrows `data` as `x` did.
        // This means we cannot access `data` until the guard goes away.
        let guard = thread::scoped(move || {
            *x *= 2;
        });
        // store the thread's guard for later
        guards.push(guard);
    }
    // All guards are dropped here, forcing the threads to join
    // (this thread blocks here until the others terminate).
    // Once the threads join, the borrow expires and the data becomes
    // accessible again in this thread.
}
// data is definitely mutated here.
```

In principle, this totally works! Rust's ownership system perfectly ensures it!
...except it relies on a destructor being called to be safe.

```rust,ignore
let mut data = Box::new(0);
{
    let guard = thread::scoped(|| {
        // This is at best a data race. At worst, it's also a use-after-free.
        *data += 1;
    });
    // Because the guard is forgotten, expiring the loan without blocking this
    // thread.
    mem::forget(guard);
}
// So the Box is dropped here while the scoped thread may or may not be trying
// to access it.
```

Dang. Here the destructor running was pretty fundamental to the API, and it had
to be scrapped in favor of a completely different design.
