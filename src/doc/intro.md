% A 30-minute Introduction to Rust

Rust is a systems programming language that focuses on strong compile-time correctness guarantees.
It improves upon the ideas other systems languages like C++, D,
and Cyclone by providing very strong guarantees and explicit control over the life cycle of memory.
Strong memory guarantees make writing correct concurrent Rust code easier than in other languages.
This might sound very complex, but it's easier than it sounds!
This tutorial will give you an idea of what Rust is like in about thirty minutes.
It expects that you're at least vaguely familiar with a previous 'curly brace' language.
The concepts are more important than the syntax,
so don't worry if you don't get every last detail:
the [tutorial](http://static.rust-lang.org/doc/master/tutorial.html) can help you out with that later.

Let's talk about the most important concept in Rust, "ownership,"
and its implications on a task that programmers usually find very difficult: concurrency.

## Ownership

Ownership is central to Rust,
and is one of its more interesting and unique features.
"Ownership" refers to which parts of your code are allowed to modify various parts of memory.
Let's start by looking at some C++ code:

```
int *dangling(void)
{
    int i = 1234;
    return &i;
}

int add_one(void)
{
    int *num = dangling();
    return *num + 1;
}
```

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
More modern C++ uses RAII with constructors/destructors,
but it amounts to the same thing.
This problem is called a 'dangling pointer,'
and it's not possible to write Rust code that has it.
Let's try:

```
fn dangling() -> &int {
    let i = 1234;
    return &i;
}

fn add_one() -> int {
    let num = dangling();
    return *num + 1;
}
```

When you try to compile this program, you'll get an interesting (and long) error message:

```
temp.rs:3:11: 3:13 error: borrowed value does not live long enough
temp.rs:3     return &i;

temp.rs:1:22: 4:1 note: borrowed pointer must be valid for the anonymous lifetime #1 defined on the block at 1:22...
temp.rs:1 fn dangling() -> &int {
temp.rs:2     let i = 1234;
temp.rs:3     return &i;
temp.rs:4 }
                            
temp.rs:1:22: 4:1 note: ...but borrowed value is only valid for the block at 1:22
temp.rs:1 fn dangling() -> &int {      
temp.rs:2     let i = 1234;            
temp.rs:3     return &i;               
temp.rs:4  }                            
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
the code "borrows" it.
It borrows it for a certain period of time, called a "lifetime."

That's all there is to it.
That doesn't seem so hard, right?
Let's go back to that error message:
`error: borrowed value does not live long enough`.
We tried to loan out a particular variable, `i`,
using Rust's borrowed pointers: the `&`.
But Rust knew that the variable would be invalid after the function returns,
and so it tells us that:
`borrowed pointer must be valid for the anonymous lifetime #1... but borrowed value is only valid for the block`.
Neat!

That's a great example for stack memory,
but what about heap memory?
Rust has a second kind of pointer,
a 'unique' pointer,
that you can create with a `~`.
Check it out:

```
fn dangling() -> ~int {
    let i = ~1234;
    return i;
}

fn add_one() -> int {
    let num = dangling();
    return *num + 1;
}
```

This code will successfully compile.
Note that instead of a stack allocated `1234`,
we use an owned pointer to that value instead: `~1234`.
You can roughly compare these two lines:

```
// rust
let i = ~1234;

// C++
int *i = new int;
*i = 1234;
```

Rust is able to infer the size of the type,
then allocates the correct amount of memory and sets it to the value you asked for.
This means that it's impossible to allocate uninitialized memory:
Rust does not have the concept of null.
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

You've seen one way that ownership and lifetimes are useful to prevent code that would normally be dangerous in a less-strict language,
but let's talk about another: concurrency.

## Concurrency

Concurrency is an incredibly hot topic in the software world right now.
It's always been an interesting area of study for computer scientists,
but as usage of the Internet explodes,
people are looking to improve the number of users a given service can handle.
Concurrency is one way of achieving this goal.
There is a pretty big drawback to concurrent code, though:
it can be hard to reason about,
because it is non-deterministic.
There are a few different approaches to writing good concurrent code,
but let's talk about how Rust's notions of ownership and lifetimes can assist with achieving correct but concurrent code.

First, let's go over a simple concurrency example in Rust.
Rust allows you to spin up 'tasks,'
which are lightweight, 'green' threads.
These tasks do not have any shared memory, and so,
we communicate between tasks with a 'channel'.
Like this:

```
fn main() {
    let numbers = [1,2,3];

    let (port, chan)  = Chan::new();
    chan.send(numbers);

    do spawn {
        let numbers = port.recv();
        println!("{:d}", numbers[0]);
    }
}
```

In this example, we create a vector of numbers.
We then make a new `Chan`,
which is the name of the package Rust implements channels with.
This returns two different ends of the channel:
a channel and a port.
You send data into the channel end, and it comes out the port end.
The `spawn` function spins up a new task.
As you can see in the code,
we call `port.recv()` (short for 'receive') inside of the new task,
and we call `chan.send()` outside,
passing in our vector.
We then print the first element of the vector.

This works out because Rust copies the vector when it is sent through the channel.
That way, if it were mutable, there wouldn't be a race condition.
However, if we're making a lot of tasks, or if our data is very large,
making a copy for each task inflates our memory usage with no real benefit.

Enter Arc.
Arc stands for 'atomically reference counted,'
and it's a way to share immutable data between multiple tasks.
Here's some code:

```
extern mod extra;
use extra::arc::Arc;

fn main() {
    let numbers = [1,2,3];

    let numbers_arc = Arc::new(numbers);

    for num in range(0, 3) {
        let (port, chan)  = Chan::new();
        chan.send(numbers_arc.clone());

        do spawn {
            let local_arc = port.recv();
            let task_numbers = local_arc.get();
            println!("{:d}", task_numbers[num]);
        }
    }
}
```

This is very similar to the code we had before,
except now we loop three times,
making three tasks,
and sending an `Arc` between them.
`Arc::new` creates a new Arc,
`.clone()` makes a new reference to that Arc,
and `.get()` gets the value out of the Arc.
So we make a new reference for each task,
send that reference down the channel,
and then use the reference to print out a number.
Now we're not copying our vector.

Arcs are great for immutable data,
but what about mutable data?
Shared mutable state is the bane of the concurrent programmer.
You can use a mutex to protect shared mutable state,
but if you forget to acquire the mutex, bad things can happen.

Rust provides a tool for shared mutable state: `RWArc`.
This variant of an Arc allows the contents of the Arc to be mutated.
Check it out:

```
extern mod extra;
use extra::arc::RWArc;

fn main() {
    let numbers = [1,2,3];

    let numbers_arc = RWArc::new(numbers);

    for num in range(0, 3) {
        let (port, chan)  = Chan::new();
        chan.send(numbers_arc.clone());

        do spawn {
            let local_arc = port.recv();

            local_arc.write(|nums| {
                nums[num] += 1
            });

            local_arc.read(|nums| {
                println!("{:d}", nums[num]);
            })
        }
    }
}
```

We now use the `RWArc` package to get a read/write Arc.
The read/write Arc has a slightly different API than `Arc`:
`read` and `write` allow you to, well, read and write the data.
They both take closures as arguments,
and the read/write Arc will, in the case of write,
acquire a mutex,
and then pass the data to this closure.
After the closure does its thing, the mutex is released.

You can see how this makes it impossible to mutate the state without remembering to aquire the lock.
We gain the efficiency of shared mutable state,
while retaining the safety of disallowing shared mutable state.

But wait, how is that possible?
We can't both allow and disallow mutable state.
What gives?

## A footnote: unsafe

So, the Rust language does not allow for shared mutable state,
yet I just showed you some code that has it.
How's this possible? The answer: `unsafe`.

You see, while the Rust compiler is very smart,
and saves you from making mistakes you might normally make,
it's not an artificial intelligence.
Because we're smarter than the compiler,
sometimes, we need to over-ride this safe behavior.
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
such as doing FFI into a C library,
performance (in certain cases),
and to provide a safe abstraction around operations that normally would not be safe.
Our Arcs are an example of this last purpose.
We can safely hand out multiple references to the `Arc`,
because we are sure the data is immutable,
and therefore it is safe to share.
We can hand out multiple references to the `RWArc`,
because we know that we've wrapped the data in a mutex,
and therefore it is safe to share.
But the Rust compiler can't know that we've made these choices,
so _inside_ the implementation of the Arcs,
we use `unsafe` blocks to do (normally) dangerous things.
But we expose a safe interface,
which means that the Arcs are impossible to use incorrectly.

This is how Rust's type system allows you to not make some of the mistakes that make concurrent programming difficult,
yet get the efficiency of languages such as C++.

## That's all, folks

I hope that this taste of Rust has given you an idea if Rust is the right language for you.
If that's true,
I encourage you to check out [the tutorial](http://static.rust-lang.org/doc/0.9/tutorial.html) for a full,
in-depth exploration of Rust's syntax and concepts.