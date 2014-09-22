% A 30-minute Introduction to Rust

Rust is a systems programming language that combines strong compile-time correctness guarantees with fast performance.
It improves upon the ideas of other systems languages like C++
by providing guaranteed memory safety (no crashes, no data races) and complete control over the lifecycle of memory.
Strong memory guarantees make writing correct concurrent Rust code easier than in other languages.
This introduction will give you an idea of what Rust is like in about thirty minutes.
It expects that you're at least vaguely familiar with a previous 'curly brace' language,
but does not require prior experience with systems programming.
The concepts are more important than the syntax,
so don't worry if you don't get every last detail:
the [guide](guide.html) can help you out with that later.

Let's talk about the most important concept in Rust, "ownership,"
and its implications on a task that programmers usually find very difficult: concurrency.

# The power of ownership

Ownership is central to Rust,
and is the feature from which many of Rust's powerful capabilities are derived.
"Ownership" refers to which parts of your code are allowed to read,
write, and ultimately release, memory.
Let's start by looking at some C++ code:

```cpp
int* dangling(void)
{
    int i = 1234;
    return &i;
}

int add_one(void)
{
    int* num = dangling();
    return *num + 1;
}
```

**Note: The above C++ code is deliberately simple and non-idiomatic for the purpose
of demonstration. It is not representative of production-quality C++ code.**

This function allocates an integer on the stack,
and stores it in a variable, `i`.
It then returns a reference to the variable `i`.
There's just one problem:
stack memory becomes invalid when the function returns.
This means that in the second line of `add_one`,
`num` points to some garbage values,
and we won't get the effect that we want.
While this is a trivial example,
it can happen quite often in C++ code.
There's a similar problem when memory on the heap is allocated with `malloc` (or `new`),
then freed with `free` (or `delete`),
yet your code attempts to do something with the pointer to that memory.
This problem is called a 'dangling pointer,'
and it's not possible to write Rust code that has it.
Let's try writing it in Rust:

```ignore
fn dangling() -> &int {
    let i = 1234;
    return &i;
}

fn add_one() -> int {
    let num = dangling();
    return *num + 1;
}

fn main() {
    add_one();
}
```

Save this program as `dangling.rs`. When you try to compile this program with `rustc dangling.rs`, you'll get an interesting (and long) error message:

```text
dangling.rs:3:12: 3:14 error: `i` does not live long enough
dangling.rs:3     return &i;
                         ^~
dangling.rs:1:23: 4:2 note: reference must be valid for the anonymous lifetime #1 defined on the block at 1:22...
dangling.rs:1 fn dangling() -> &int {
dangling.rs:2     let i = 1234;
dangling.rs:3     return &i;
dangling.rs:4 }
dangling.rs:1:23: 4:2 note: ...but borrowed value is only valid for the block at 1:22
dangling.rs:1 fn dangling() -> &int {
dangling.rs:2     let i = 1234;
dangling.rs:3     return &i;
dangling.rs:4 }
error: aborting due to previous error
```

In order to fully understand this error message,
we need to talk about what it means to "own" something.
So for now,
let's just accept that Rust will not allow us to write code with a dangling pointer,
and we'll come back to this code once we understand ownership.

Let's forget about programming for a second and talk about books.
I like to read physical books,
and sometimes I really like one and tell my friends they should read it.
While I'm reading my book, I own it: the book is in my possession.
When I loan the book out to someone else for a while, they "borrow" it from me.
And when you borrow a book, it's yours for a certain period of time,
and then you give it back to me, and I own it again. Right?

This concept applies directly to Rust code as well:
some code "owns" a particular pointer to memory.
It's the sole owner of that pointer.
It can also lend that memory out to some other code for a while:
that code "borrows" the memory,
and it borrows it for a precise period of time,
called a "lifetime."

That's all there is to it.
That doesn't seem so hard, right?
Let's go back to that error message:
`error: 'i' does not live long enough`.
We tried to loan out a particular variable, `i`,
using a reference (the `&` operator) but Rust knew that the variable would be invalid after the function returns,
and so it tells us that:
`reference must be valid for the anonymous lifetime #1...`.
Neat!

That's a great example for stack memory,
but what about heap memory?
Rust has a second kind of pointer,
an 'owned box',
that you can create with the `box` operator.
Check it out:

```

fn dangling() -> Box<int> {
    let i = box 1234i;
    return i;
}

fn add_one() -> int {
    let num = dangling();
    return *num + 1;
}
```

Now instead of a stack allocated `1234i`,
we have a heap allocated `box 1234i`.
Whereas `&` borrows a pointer to existing memory,
creating an owned box allocates memory on the heap and places a value in it,
giving you the sole pointer to that memory.
You can roughly compare these two lines:

```
// Rust
let i = box 1234i;
```

```cpp
// C++
int *i = new int;
*i = 1234;
```

Rust infers the correct type,
allocates the correct amount of memory and sets it to the value you asked for.
This means that it's impossible to allocate uninitialized memory:
*Rust does not have the concept of null*.
Hooray!
There's one other difference between this line of Rust and the C++:
The Rust compiler also figures out the lifetime of `i`,
and then inserts a corresponding `free` call after it's invalid,
like a destructor in C++.
You get all of the benefits of manually allocated heap memory without having to do all the bookkeeping yourself.
Furthermore, all of this checking is done at compile time,
so there's no runtime overhead.
You'll get (basically) the exact same code that you'd get if you wrote the correct C++,
but it's impossible to write the incorrect version, thanks to the compiler.

You've seen one way that ownership and borrowing are useful to prevent code that would normally be dangerous in a less-strict language,
but let's talk about another: concurrency.

# Owning concurrency

Concurrency is an incredibly hot topic in the software world right now.
It's always been an interesting area of study for computer scientists,
but as usage of the Internet explodes,
people are looking to improve the number of users a given service can handle.
Concurrency is one way of achieving this goal.
There is a pretty big drawback to concurrent code, though:
it can be hard to reason about, because it is non-deterministic.
There are a few different approaches to writing good concurrent code,
but let's talk about how Rust's notions of ownership and lifetimes contribute to correct but concurrent code.

First, let's go over a simple concurrency example.
Rust makes it easy to create "tasks",
otherwise known as "threads".
Typically, tasks do not share memory but instead communicate amongst each other with 'channels', like this:

```
fn main() {
    let numbers = vec![1i, 2i, 3i];

    let (tx, rx)  = channel();
    tx.send(numbers);

    spawn(proc() {
        let numbers = rx.recv();
        println!("{}", numbers[0]);
    })
}
```

In this example, we create a boxed array of numbers.
We then make a 'channel',
Rust's primary means of passing messages between tasks.
The `channel` function returns two different ends of the channel:
a `Sender` and `Receiver` (commonly abbreviated `tx` and `rx`).
The `spawn` function spins up a new task,
given a *heap allocated closure* to run.
As you can see in the code,
we call `tx.send()` from the original task,
passing in our boxed array,
and we call `rx.recv()` (short for 'receive') inside of the new task:
values given to the `Sender` via the `send` method come out the other end via the `recv` method on the `Receiver`.

Now here's the exciting part:
because `numbers` is an owned type,
when it is sent across the channel,
it is actually *moved*,
transferring ownership of `numbers` between tasks.
This ownership transfer is *very fast* -
in this case simply copying a pointer -
while also ensuring that the original owning task cannot create data races by continuing to read or write to `numbers` in parallel with the new owner.

To prove that Rust performs the ownership transfer,
try to modify the previous example to continue using the variable `numbers`:

```ignore
fn main() {
    let numbers = vec![1i, 2i, 3i];

    let (tx, rx)  = channel();
    tx.send(numbers);

    spawn(proc() {
        let numbers = rx.recv();
        println!("{}", numbers[0]);
    });

    // Try to print a number from the original task
    println!("{}", numbers[0]);
}
```

The compiler will produce an error indicating that the value is no longer in scope:

```text
concurrency.rs:12:20: 12:27 error: use of moved value: 'numbers'
concurrency.rs:12     println!("{}", numbers[0]);
                                     ^~~~~~~
```

Since only one task can own a boxed array at a time,
if instead of distributing our `numbers` array to a single task we wanted to distribute it to many tasks,
we would need to copy the array for each.
Let's see an example that uses the `clone` method to create copies of the data:

```
fn main() {
    let numbers = vec![1i, 2i, 3i];

    for num in range(0u, 3) {
        let (tx, rx)  = channel();
        // Use `clone` to send a *copy* of the array
        tx.send(numbers.clone());

        spawn(proc() {
            let numbers = rx.recv();
            println!("{:d}", numbers[num]);
        })
    }
}
```

This is similar to the code we had before,
except now we loop three times,
making three tasks,
and *cloning* `numbers` before sending it.

However, if we're making a lot of tasks,
or if our data is very large,
creating a copy for each task requires a lot of work and a lot of extra memory for little benefit.
In practice, we might not want to do this because of the cost.
Enter `Arc`,
an atomically reference counted box ("A.R.C." == "atomically reference counted").
`Arc` is the most common way to *share* data between tasks.
Here's some code:

```
use std::sync::Arc;

fn main() {
    let numbers = Arc::new(vec![1i, 2i, 3i]);

    for num in range(0u, 3) {
        let (tx, rx)  = channel();
        tx.send(numbers.clone());

        spawn(proc() {
            let numbers = rx.recv();
            println!("{:d}", (*numbers)[num as uint]);
        })
    }
}
```

This is almost exactly the same,
except that this time `numbers` is first put into an `Arc`.
`Arc::new` creates the `Arc`,
`.clone()` makes another `Arc` that refers to the same contents.
So we clone the `Arc` for each task,
send that clone down the channel,
and then use it to print out a number.
Now instead of copying an entire array to send it to our multiple tasks we are just copying a pointer (the `Arc`) and *sharing* the array.

How can this work though?
Surely if we're sharing data then can't we cause data races if one task writes to the array while others read?

Well, Rust is super-smart and will only let you put data into an `Arc` that is provably safe to share.
In this case, it's safe to share the array *as long as it's immutable*,
i.e. many tasks may read the data in parallel as long as none can write.
So for this type and many others `Arc` will only give you an immutable view of the data.

Arcs are great for immutable data,
but what about mutable data?
Shared mutable state is the bane of the concurrent programmer:
you can use a mutex to protect shared mutable state,
but if you forget to acquire the mutex, bad things can happen, including crashes.
Rust provides mutexes but makes it impossible to use them in a way that subverts memory safety.

Let's take the same example yet again,
and modify it to mutate the shared state:

```
use std::sync::{Arc, Mutex};

fn main() {
    let numbers_lock = Arc::new(Mutex::new(vec![1i, 2i, 3i]));

    for num in range(0u, 3) {
        let (tx, rx)  = channel();
        tx.send(numbers_lock.clone());

        spawn(proc() {
            let numbers_lock = rx.recv();

            // Take the lock, along with exclusive access to the underlying array
            let mut numbers = numbers_lock.lock();

            // This is ugly for now because of the need for `get_mut`, but
            // will be replaced by `numbers[num as uint] += 1`
            // in the near future.
            // See: https://github.com/rust-lang/rust/issues/6515
            *numbers.get_mut(num as uint) += 1;

            println!("{}", (*numbers)[num as uint]);

            // When `numbers` goes out of scope the lock is dropped
        })
    }
}
```

This example is starting to get more subtle,
but it hints at the powerful composability of Rust's concurrent types.
This time we've put our array of numbers inside a `Mutex` and then put *that* inside the `Arc`.
Like immutable data,
`Mutex`es are sharable,
but unlike immutable data,
data inside a `Mutex` may be mutated as long as the mutex is locked.

The `lock` method here returns not your original array or a pointer thereof,
but a `MutexGuard`,
a type that is responsible for releasing the lock when it goes out of scope.
This same `MutexGuard` can transparently be treated as if it were the value the `Mutex` contains,
as you can see in the subsequent indexing operation that performs the mutation.

OK, let's stop there before we get too deep.

# A footnote: unsafe

The Rust compiler and libraries are entirely written in Rust;
we say that Rust is "self-hosting".
If Rust makes it impossible to unsafely share data between threads,
and Rust is written in Rust,
then how does it implement concurrent types like `Arc` and `Mutex`?
The answer: `unsafe`.

You see, while the Rust compiler is very smart,
and saves you from making mistakes you might normally make,
it's not an artificial intelligence.
Because we're smarter than the compiler -
sometimes - we need to over-ride this safe behavior.
For this purpose, Rust has an `unsafe` keyword.
Within an `unsafe` block,
Rust turns off many of its safety checks.
If something bad happens to your program,
you only have to audit what you've done inside `unsafe`,
and not the entire program itself.

If one of the major goals of Rust was safety,
why allow that safety to be turned off?
Well, there are really only three main reasons to do it:
interfacing with external code,
such as doing FFI into a C library;
performance (in certain cases);
and to provide a safe abstraction around operations that normally would not be safe.
Our `Arc`s are an example of this last purpose.
We can safely hand out multiple pointers to the contents of the `Arc`,
because we are sure the data is safe to share.
But the Rust compiler can't know that we've made these choices,
so _inside_ the implementation of the Arcs,
we use `unsafe` blocks to do (normally) dangerous things.
But we expose a safe interface,
which means that the `Arc`s are impossible to use incorrectly.

This is how Rust's type system prevents you from making some of the mistakes that make concurrent programming difficult,
yet get the efficiency of languages such as C++.

# That's all, folks

I hope that this taste of Rust has given you an idea if Rust is the right language for you.
If that's true,
I encourage you to check out [the guide](guide.html) for a full,
in-depth exploration of Rust's syntax and concepts.
