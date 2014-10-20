% A 30-minute Introduction to Rust

Rust is a modern systems programming language focusing on safety and speed. It
accomplishes these goals by being memory safe without using garbage collection.

This introduction will give you a rough idea of what Rust is like, eliding many
details. It does not require prior experience with systems programming, but you
may find the syntax easier if you've used a 'curly brace' programming language
before, like C or JavaScript. The concepts are more important than the syntax,
so don't worry if you don't get every last detail: you can read [the
Guide](guide.html) to get a more complete explanation.

Because this is about high-level concepts, you don't need to actually install
Rust to follow along. If you'd like to anyway, check out [the
homepage](http://rust-lang.org) for explanation.

To show off Rust, let's talk about how easy it is to get started with Rust.
Then, we'll talk about Rust's most interesting feature, **ownership**, and
then discuss how it makes concurrency easier to reason about. Finally,
we'll talk about how Rust breaks down the perceived dichotomy between speed
and safety.

# Tools

Getting started on a new Rust project is incredibly easy, thanks to Rust's
package manager, [Cargo](http://crates.io).

To start a new project with Cargo, use `cargo new`:

```{bash}
$ cargo new hello_world --bin
```

We're passing `--bin` because we're making a binary program: if we
were making a library, we'd leave it off.

Let's check out what Cargo has generated for us:

```{bash}
$ cd hello_world
$ tree .
.
├── Cargo.toml
└── src
    └── main.rs

1 directory, 2 files
```

This is all we need to get started. First, let's check out `Cargo.toml`:

```{toml}
[package]

name = "hello_world"
version = "0.0.1"
authors = ["Your Name <you@example.com>"]
```

This is called a **manifest**, and it contains all of the metadata that Cargo
needs to compile your project. 

Here's what's in `src/main.rs`:

```{rust}
fn main() {
    println!("Hello, world!")
}
```

Cargo generated a 'hello world' for us. We'll talk more about the syntax here
later, but that's what Rust code looks like! Let's compile and run it:

```{bash}
$ cargo run
   Compiling hello_world v0.0.1 (file:///Users/you/src/hello_world)
     Running `target/hello_world`
Hello, world!
```

Using an external dependency in Rust is incredibly easy. You add a line to
your `Cargo.toml`:

```{toml}
[package]

name = "hello_world"
version = "0.0.1"
authors = ["Your Name <someone@example.com>"]

[dependencies.semver]

git = "https://github.com/rust-lang/semver.git"
```

You added the `semver` library, which parses version numbers and compares them
according to the [SemVer specification](http://semver.org/).

Now, you can pull in that library using `extern crate` in
`main.rs`.

```{rust,ignore}
extern crate semver;

use semver::Version;

fn main() {
    assert!(Version::parse("1.2.3") == Ok(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(),
        build: vec!(),
    }));

    println!("Versions compared successfully!");
}
```

Again, we'll discuss the exact details of all of this syntax soon. For now,
let's compile and run it:

```{bash}
$ cargo run
    Updating git repository `https://github.com/rust-lang/semver.git`
   Compiling semver v0.0.1 (https://github.com/rust-lang/semver.git#bf739419)
   Compiling hello_world v0.0.1 (file:///home/you/projects/hello_world)
     Running `target/hello_world`
Versions compared successfully!
```

Because we only specified a repository without a version, if someone else were
to try out our project at a later date, when `semver` was updated, they would
get a different, possibly incompatible version. To solve this problem, Cargo
produces a file, `Cargo.lock`, which records the versions of any dependencies.
This gives us repeatable builds.

There is a lot more here, and this is a whirlwind tour, but you should feel
right at home if you've used tools like [Bundler](http://bundler.io/),
[npm](https://www.npmjs.org/), or [pip](https://pip.pypa.io/en/latest/).
There's no `Makefile`s or endless `autotools` output here. (Rust's tooling does
[play nice with external libraries written in those
tools](http://crates.io/native-build.html), if you need to.)

Enough about tools, let's talk code!

# Ownership

Rust's defining feature is 'memory safety without garbage collection.' Let's
take a moment to talk about what that means. **Memory safety** means that the
programming language eliminates certain kinds of bugs, such as [buffer
overflows](http://en.wikipedia.org/wiki/Buffer_overflow) and [dangling
pointers](http://en.wikipedia.org/wiki/Dangling_pointer). These problems occur
when you have unrestricted access to memory. As an example, here's some Ruby
code:

```{ruby}
v = [];

v.push("Hello");

x = v[0];

v.push("world");

puts x
```

We make an array, `v`, and then call `push` on it. `push` is a method which
adds an element to the end of an array.

Next, we make a new variable, `x`, that's equal to the first element of
the array. Simple, but this is where the 'bug' will appear.

Let's keep going. We then call `push` again, pushing "world" onto the
end of the array. `v` now is `["Hello", "world"]`.

Finally, we print `x` with the `puts` method. This prints "Hello."

All good? Let's go over a similar, but subtly different example, in C++:

```{cpp}
#include<iostream>
#include<vector>
#include<string>

int main() {
    std::vector<std::string> v;

    v.push_back("Hello");

    std::string& x = v[0];

    v.push_back("world");

    std::cout << x;
}
```

It's a little more verbose due to the static typing, but it's almost the same
thing. We make a `std::vector` of `std::string`s, we call `push_back` (same as
`push`) on it, take a reference to the first element of the vector, call
`push_back` again, and then print out the reference.

There's two big differences here: one, they're not _exactly_ the same thing,
and two...

```{bash}
$ g++ hello.cpp -Wall -Werror
$ ./a.out 
Segmentation fault (core dumped)
```

A crash! (Note that this is actually system-dependent. Because referring to an
invalid reference is undefined behavior, the compiler can do anything,
including the right thing!) Even though we compiled with flags to give us as
many warnings as possible, and to treat those warnings as errors, we got no
errors. When we ran the program, it crashed.

Why does this happen? When we prepend to an array, its length changes. Since
its length changes, we may need to allocate more memory. In Ruby, this happens
as well, we just don't think about it very often. So why does the C++ version
segfault when we allocate more memory?

The answer is that in the C++ version, `x` is a **reference** to the memory
location where the first element of the array is stored. But in Ruby, `x` is a
standalone value, not connected to the underyling array at all. Let's dig into
the details for a moment. Your program has access to memory, provided to it by
the operating system. Each location in memory has an address.  So when we make
our vector, `v`, it's stored in a memory location somewhere:

| location | name | value |
|----------|------|-------|
| 0x30     | v    |       |

(Address numbers made up, and in hexadecimal. Those of you with deep C++
knowledge, there are some simplifications going on here, like the lack of an
allocated length for the vector. This is an introduction.)

When we push our first string onto the array, we allocate some memory,
and `v` refers to it:

| location | name | value    |
|----------|------|----------|
| 0x30     | v    | 0x18     |
| 0x18     |      | "Hello"  |

We then make a reference to that first element. A reference is a variable
that points to a memory location, so its value is the memory location of
the `"Hello"` string:

| location | name | value    |
|----------|------|----------|
| 0x30     | v    | 0x18     |
| 0x18     |      | "Hello"  |
| 0x14     | x    | 0x18     |

When we push `"world"` onto the vector with `push_back`, there's no room:
we only allocated one element. So, we need to allocate two elements,
copy the `"Hello"` string over, and update the reference. Like this:

| location | name | value    |
|----------|------|----------|
| 0x30     | v    | 0x08     |
| 0x18     |      | GARBAGE  |
| 0x14     | x    | 0x18     |
| 0x08     |      | "Hello"  |
| 0x04     |      | "world"  |

Note that `v` now refers to the new list, which has two elements. It's all
good. But our `x` didn't get updated! It still points at the old location,
which isn't valid anymore. In fact, [the documentation for `push_back` mentions
this](http://en.cppreference.com/w/cpp/container/vector/push_back):

> If the new `size()` is greater than `capacity()` then all iterators and
> references (including the past-the-end iterator) are invalidated.

Finding where these iterators and references are is a difficult problem, and
even in this simple case, `g++` can't help us here. While the bug is obvious in
this case, in real code, it can be difficult to track down the source of the
error.

Before we talk about this solution, why didn't our Ruby code have this problem?
The semantics are a little more complicated, and explaining Ruby's internals is
out of the scope of a guide to Rust. But in a nutshell, Ruby's garbage
collector keeps track of references, and makes sure that everything works as
you might expect. This comes at an efficiency cost, and the internals are more
complex.  If you'd really like to dig into the details, [this
article](http://patshaughnessy.net/2012/1/18/seeing-double-how-ruby-shares-string-values)
can give you more information.

Garbage collection is a valid approach to memory safety, but Rust chooses a
different path.  Let's examine what the Rust version of this looks like:

```{rust,ignore}
fn main() {
    let mut v = vec![];

    v.push("Hello");

    let x = &v[0];

    v.push("world");

    println!("{}", x);
}
```

This looks like a bit of both: fewer type annotations, but we do create new
variables with `let`. The method name is `push`, some other stuff is different,
but it's pretty close. So what happens when we compile this code?  Does Rust
print `"Hello"`, or does Rust crash?

Neither. It refuses to compile:

```{notrust,ignore}
$ cargo run
   Compiling hello_world v0.0.1 (file:///Users/you/src/hello_world)
main.rs:8:5: 8:6 error: cannot borrow `v` as mutable because it is also borrowed as immutable
main.rs:8     v.push("world");
              ^
main.rs:6:14: 6:15 note: previous borrow of `v` occurs here; the immutable borrow prevents subsequent moves or mutable borrows of `v` until the borrow ends
main.rs:6     let x = &v[0];
                       ^
main.rs:11:2: 11:2 note: previous borrow ends here
main.rs:1 fn main() {
...
main.rs:11 }
           ^
error: aborting due to previous error
```

When we try to mutate the array by `push`ing it the second time, Rust throws
an error. It says that we "cannot borrow v as mutable because it is also
borrowed as immutable." What's up with "borrowed"?

In Rust, the type system encodes the notion of **ownership**. The variable `v`
is an "owner" of the vector. When we make a reference to `v`, we let that
variable (in this case, `x`) 'borrow' it for a while. Just like if you own a
book, and you lend it to me, I'm borrowing the book.

So, when I try to modify the vector with the second call to `push`, I need
to be owning it. But `x` is borrowing it. You can't modify something that
you've lent to someone. And so Rust throws an error.

So how do we fix this problem? Well, we can make a copy of the element:


```{rust}
fn main() {
    let mut v = vec![];

    v.push("Hello");

    let x = v[0].clone();

    v.push("world");

    println!("{}", x);
}
```

Note the addition of `clone()`. This creates a copy of the element, leaving
the original untouched. Now, we no longer have two references to the same
memory, and so the compiler is happy. Let's give that a try:

```{bash}
$ cargo run
   Compiling hello_world v0.0.1 (file:///Users/you/src/hello_world)
     Running `target/hello_world`
Hello
```

Same result. Now, making a copy can be inefficient, so this solution may not be
acceptable. There are other ways to get around this problem, but this is a toy
example, and because we're in an introduction, we'll leave that for later.

The point is, the Rust compiler and its notion of ownership has saved us from a
bug that would crash the program. We've achieved safety, at compile time,
without needing to rely on a garbage collector to handle our memory.

# Concurrency

Rust's ownership model can help in other ways, as well. For example, take
concurrency. Concurrency is a big topic, and an important one for any modern
programming language. Let's take a look at how ownership can help you write
safe concurrent programs.

Here's an example of a concurrent Rust program:

```{rust}
fn main() {
    for _ in range(0u, 10u) {
        spawn(proc() {
            println!("Hello, world!");
        });
    }
}
```

This program creates ten threads, who all print `Hello, world!`. The `spawn`
function takes one argument, a `proc`. 'proc' is short for 'procedure,' and is
a form of closure. This closure is executed in a new thread, created by `spawn`
itself.

One common form of problem in concurrent programs is a 'data race.' This occurs
when two different threads attempt to access the same location in memory in a
non-synchronized way, where at least one of them is a write. If one thread is
attempting to read, and one thread is attempting to write, you cannot be sure
that your data will not be corrupted. Note the first half of that requirement:
two threads that attempt to access the same location in memory. Rust's
ownership model can track which pointers own which memory locations, which
solves this problem.

Let's see an example. This Rust code will not compile:

```{rust,ignore}
fn main() {
    let mut numbers = vec![1i, 2i, 3i];

    for i in range(0u, 3u) {
        spawn(proc() {
            for j in range(0, 3) { numbers[j] += 1 }
        });
    }
}
```

It gives us this error:

```{notrust,ignore}
6:71 error: capture of moved value: `numbers`
    for j in range(0, 3) { numbers[j] += 1 }
               ^~~~~~~
7:50 note: `numbers` moved into closure environment here because it has type `proc():Send`, which is non-copyable (perhaps you meant to use clone()?)
    spawn(proc() {
        for j in range(0, 3) { numbers[j] += 1 }
    });
6:79 error: cannot assign to immutable dereference (dereference is implicit, due to indexing)
        for j in range(0, 3) { numbers[j] += 1 }
                           ^~~~~~~~~~~~~~~
```

It mentions that "numbers moved into closure environment". Because we referred
to `numbers` inside of our `proc`, and we create three `proc`s, we would have
three references. Rust detects this and gives us the error: we claim that
`numbers` has ownership, but our code tries to make three owners. This may
cause a safety problem, so Rust disallows it.

What to do here? Rust has two types that helps us: `Arc<T>` and `Mutex<T>`.
"Arc" stands for "atomically reference counted." In other words, an Arc will
keep track of the number of references to something, and not free the
associated resource until the count is zero. The 'atomic' portion refers to an
Arc's usage of concurrency primitives to atomically update the count, making it
safe across threads. If we use an Arc, we can have our three references. But,
an Arc does not allow mutable borrows of the data it holds, and we want to
modify what we're sharing. In this case, we can use a `Mutex<T>` inside of our
Arc. A Mutex will synchronize our accesses, so that we can ensure that our
mutation doesn't cause a data race.

Here's what using an Arc with a Mutex looks like:

```{rust}
use std::sync::{Arc,Mutex};

fn main() {
    let numbers = Arc::new(Mutex::new(vec![1i, 2i, 3i]));

    for i in range(0u, 3u) {
        let number = numbers.clone();
        spawn(proc() {
            let mut array = number.lock();

            (*(*array).get_mut(i)) += 1;

            println!("numbers[{}] is {}", i, (*array)[i]);
        });
    }
}
```

We first have to `use` the appropriate library, and then we wrap our vector in
an Arc with the call to `Arc::new()`. Inside of the loop, we make a new
reference to the Arc with the `clone()` method. This will increment the
reference count. When each new `numbers` variable binding goes out of scope, it
will decrement the count. The `lock()` call will return us a reference to the
value inside the Mutex, and block any other calls to `lock()` until said
reference goes out of scope.

We can compile and run this program without error, and in fact, see the
non-deterministic aspect:

```{shell}
$ cargo run
   Compiling hello_world v0.0.1 (file:///Users/you/src/hello_world)
     Running `target/hello_world`
numbers[1] is 2
numbers[0] is 1
numbers[2] is 3
$ cargo run
     Running `target/hello_world`
numbers[2] is 3
numbers[1] is 2
numbers[0] is 1
```

Each time, we get a slightly different output, because each thread works in a
different order. You may not get the same output as this sample, even.

The important part here is that the Rust compiler was able to use ownership to
give us assurance _at compile time_ that we weren't doing something incorrect
with regards to concurrency. In order to share ownership, we were forced to be
explicit and use a mechanism to ensure that it would be properly handled.

# Safety _and_ speed

Safety and speed are always presented as a continuum. On one hand, you have
maximum speed, but no safety. On the other, you have absolute safety, with no
speed. Rust seeks to break out of this mode by introducing safety at compile
time, ensuring that you haven't done anything wrong, while compiling to the
same low-level code you'd expect without the safety.

As an example, Rust's ownership system is _entirely_ at compile time. The
safety check that makes this an error about moved values:

```{rust,ignore}
fn main() {
    let vec = vec![1i, 2, 3];

    for i in range(1u, 3) {
        spawn(proc() {
            println!("{}", vec[i]);
        });
    }
}
```

carries no runtime penalty. And while some of Rust's safety features do have
a run-time cost, there's often a way to write your code in such a way that
you can remove it. As an example, this is a poor way to iterate through
a vector:

```{rust}
let vec = vec![1i, 2, 3];

for i in range(1u, vec.len()) {
     println!("{}", vec[i]);
}
```

The reason is that the access of `vec[i]` does bounds checking, to ensure
that we don't try to access an invalid index. However, we can remove this
while retaining safety. The answer is iterators:

```{rust}
let vec = vec![1i, 2, 3];

for x in vec.iter() {
    println!("{}", x);
}
```

This version uses an iterator that yields each element of the vector in turn.
Because we have a reference to the element, rather than the whole vector itself,
there's no array access bounds to check.

# Learning More

I hope that this taste of Rust has given you an idea if Rust is the right
language for you. We talked about Rust's tooling, how encoding ownership into
the type system helps you find bugs, how Rust can help you write correct
concurrent code, and how you don't have to pay a speed cost for much of this
safety.

To continue your Rustic education, read [the guide](guide.html) for a more
in-depth exploration of Rust's syntax and concepts.
