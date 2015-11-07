% Dining Philosophers

For our second project, let’s look at a classic concurrency problem. It’s
called ‘the dining philosophers’. It was originally conceived by Dijkstra in
1965, but we’ll use a lightly adapted version from [this paper][paper] by Tony
Hoare in 1985.

[paper]: http://www.usingcsp.com/cspbook.pdf

> In ancient times, a wealthy philanthropist endowed a College to accommodate
> five eminent philosophers. Each philosopher had a room in which they could
> engage in their professional activity of thinking; there was also a common
> dining room, furnished with a circular table, surrounded by five chairs, each
> labelled by the name of the philosopher who was to sit in it. They sat
> anticlockwise around the table. To the left of each philosopher there was
> laid a golden fork, and in the center stood a large bowl of spaghetti, which
> was constantly replenished. A philosopher was expected to spend most of
> their time thinking; but when they felt hungry, they went to the dining
> room, sat down in their own chair, picked up their own fork on their left,
> and plunged it into the spaghetti. But such is the tangled nature of
> spaghetti that a second fork is required to carry it to the mouth. The
> philosopher therefore had also to pick up the fork on their right. When
> they were finished they would put down both their forks, get up from their
> chair, and continue thinking. Of course, a fork can be used by only one
> philosopher at a time. If the other philosopher wants it, they just have
> to wait until the fork is available again.

This classic problem shows off a few different elements of concurrency. The
reason is that it's actually slightly tricky to implement: a simple
implementation can deadlock. For example, let's consider a simple algorithm
that would solve this problem:

1. A philosopher picks up the fork on their left.
2. They then pick up the fork on their right.
3. They eat.
4. They return the forks.

Now, let’s imagine this sequence of events:

1. Philosopher 1 begins the algorithm, picking up the fork on their left.
2. Philosopher 2 begins the algorithm, picking up the fork on their left.
3. Philosopher 3 begins the algorithm, picking up the fork on their left.
4. Philosopher 4 begins the algorithm, picking up the fork on their left.
5. Philosopher 5 begins the algorithm, picking up the fork on their left.
6. ... ? All the forks are taken, but nobody can eat!

There are different ways to solve this problem. We’ll get to our solution in
the tutorial itself. For now, let’s get started modeling the problem itself.
We’ll start with the philosophers:

```rust
struct Philosopher {
    name: String,
}

impl Philosopher {
    fn new(name: &str) -> Philosopher {
        Philosopher {
            name: name.to_string(),
        }
    }
}

fn main() {
    let p1 = Philosopher::new("Judith Butler");
    let p2 = Philosopher::new("Gilles Deleuze");
    let p3 = Philosopher::new("Karl Marx");
    let p4 = Philosopher::new("Emma Goldman");
    let p5 = Philosopher::new("Michel Foucault");
}
```

Here, we make a [`struct`][struct] to represent a philosopher. For now,
a name is all we need. We choose the [`String`][string] type for the name,
rather than `&str`. Generally speaking, working with a type which owns its
data is easier than working with one that uses references.

[struct]: structs.html
[string]: strings.html

Let’s continue:

```rust
# struct Philosopher {
#     name: String,
# }
impl Philosopher {
    fn new(name: &str) -> Philosopher {
        Philosopher {
            name: name.to_string(),
        }
    }
}
```

This `impl` block lets us define things on `Philosopher` structs. In this case,
we define an ‘associated function’ called `new`. The first line looks like this:

```rust
# struct Philosopher {
#     name: String,
# }
# impl Philosopher {
fn new(name: &str) -> Philosopher {
#         Philosopher {
#             name: name.to_string(),
#         }
#     }
# }
```

We take one argument, a `name`, of type `&str`. This is a reference to another
string. It returns an instance of our `Philosopher` struct.

```rust
# struct Philosopher {
#     name: String,
# }
# impl Philosopher {
#    fn new(name: &str) -> Philosopher {
Philosopher {
    name: name.to_string(),
}
#     }
# }
```

This creates a new `Philosopher`, and sets its `name` to our `name` argument.
Not just the argument itself, though, as we call `.to_string()` on it. This
will create a copy of the string that our `&str` points to, and give us a new
`String`, which is the type of the `name` field of `Philosopher`.

Why not accept a `String` directly? It’s nicer to call. If we took a `String`,
but our caller had a `&str`, they’d have to call this method themselves. The
downside of this flexibility is that we _always_ make a copy. For this small
program, that’s not particularly important, as we know we’ll just be using
short strings anyway.

One last thing you’ll notice: we just define a `Philosopher`, and seemingly
don’t do anything with it. Rust is an ‘expression based’ language, which means
that almost everything in Rust is an expression which returns a value. This is
true of functions as well, the last expression is automatically returned. Since
we create a new `Philosopher` as the last expression of this function, we end
up returning it.

This name, `new()`, isn’t anything special to Rust, but it is a convention for
functions that create new instances of structs. Before we talk about why, let’s
look at `main()` again:

```rust
# struct Philosopher {
#     name: String,
# }
#
# impl Philosopher {
#     fn new(name: &str) -> Philosopher {
#         Philosopher {
#             name: name.to_string(),
#         }
#     }
# }
#
fn main() {
    let p1 = Philosopher::new("Judith Butler");
    let p2 = Philosopher::new("Gilles Deleuze");
    let p3 = Philosopher::new("Karl Marx");
    let p4 = Philosopher::new("Emma Goldman");
    let p5 = Philosopher::new("Michel Foucault");
}
```

Here, we create five variable bindings with five new philosophers. These are my
favorite five, but you can substitute anyone you want. If we _didn’t_ define
that `new()` function, it would look like this:

```rust
# struct Philosopher {
#     name: String,
# }
fn main() {
    let p1 = Philosopher { name: "Judith Butler".to_string() };
    let p2 = Philosopher { name: "Gilles Deleuze".to_string() };
    let p3 = Philosopher { name: "Karl Marx".to_string() };
    let p4 = Philosopher { name: "Emma Goldman".to_string() };
    let p5 = Philosopher { name: "Michel Foucault".to_string() };
}
```

That’s much noisier. Using `new` has other advantages too, but even in
this simple case, it ends up being nicer to use.

Now that we’ve got the basics in place, there’s a number of ways that we can
tackle the broader problem here. I like to start from the end first: let’s
set up a way for each philosopher to finish eating. As a tiny step, let’s make
a method, and then loop through all the philosophers, calling it:

```rust
struct Philosopher {
    name: String,
}

impl Philosopher {
    fn new(name: &str) -> Philosopher {
        Philosopher {
            name: name.to_string(),
        }
    }

    fn eat(&self) {
        println!("{} is done eating.", self.name);
    }
}

fn main() {
    let philosophers = vec![
        Philosopher::new("Judith Butler"),
        Philosopher::new("Gilles Deleuze"),
        Philosopher::new("Karl Marx"),
        Philosopher::new("Emma Goldman"),
        Philosopher::new("Michel Foucault"),
    ];

    for p in &philosophers {
        p.eat();
    }
}
```

Let’s look at `main()` first. Rather than have five individual variable
bindings for our philosophers, we make a `Vec<T>` of them instead. `Vec<T>` is
also called a ‘vector’, and it’s a growable array type. We then use a
[`for`][for] loop to iterate through the vector, getting a reference to each
philosopher in turn.

[for]: for-loops.html

In the body of the loop, we call `p.eat()`, which is defined above:

```rust,ignore
fn eat(&self) {
    println!("{} is done eating.", self.name);
}
```

In Rust, methods take an explicit `self` parameter. That’s why `eat()` is a
method, but `new` is an associated function: `new()` has no `self`. For our
first version of `eat()`, we just print out the name of the philosopher, and
mention they’re done eating. Running this program should give you the following
output:

```text
Judith Butler is done eating.
Gilles Deleuze is done eating.
Karl Marx is done eating.
Emma Goldman is done eating.
Michel Foucault is done eating.
```

Easy enough, they’re all done! We haven’t actually implemented the real problem
yet, though, so we’re not done yet!

Next, we want to make our philosophers not just finish eating, but actually
eat. Here’s the next version:

```rust
use std::thread;
use std::time::Duration;

struct Philosopher {
    name: String,
}

impl Philosopher {
    fn new(name: &str) -> Philosopher {
        Philosopher {
            name: name.to_string(),
        }
    }

    fn eat(&self) {
        println!("{} is eating.", self.name);

        thread::sleep(Duration::from_millis(1000));

        println!("{} is done eating.", self.name);
    }
}

fn main() {
    let philosophers = vec![
        Philosopher::new("Judith Butler"),
        Philosopher::new("Gilles Deleuze"),
        Philosopher::new("Karl Marx"),
        Philosopher::new("Emma Goldman"),
        Philosopher::new("Michel Foucault"),
    ];

    for p in &philosophers {
        p.eat();
    }
}
```

Just a few changes. Let’s break it down.

```rust,ignore
use std::thread;
```

`use` brings names into scope. We’re going to start using the `thread` module
from the standard library, and so we need to `use` it.

```rust,ignore
    fn eat(&self) {
        println!("{} is eating.", self.name);

        thread::sleep(Duration::from_millis(1000));

        println!("{} is done eating.", self.name);
    }
```

We now print out two messages, with a `sleep` in the middle. This will
simulate the time it takes a philosopher to eat.

If you run this program, you should see each philosopher eat in turn:

```text
Judith Butler is eating.
Judith Butler is done eating.
Gilles Deleuze is eating.
Gilles Deleuze is done eating.
Karl Marx is eating.
Karl Marx is done eating.
Emma Goldman is eating.
Emma Goldman is done eating.
Michel Foucault is eating.
Michel Foucault is done eating.
```

Excellent! We’re getting there. There’s just one problem: we aren’t actually
operating in a concurrent fashion, which is a core part of the problem!

To make our philosophers eat concurrently, we need to make a small change.
Here’s the next iteration:

```rust
use std::thread;
use std::time::Duration;

struct Philosopher {
    name: String,
}

impl Philosopher {
    fn new(name: &str) -> Philosopher {
        Philosopher {
            name: name.to_string(),
        }
    }

    fn eat(&self) {
        println!("{} is eating.", self.name);

        thread::sleep(Duration::from_millis(1000));

        println!("{} is done eating.", self.name);
    }
}

fn main() {
    let philosophers = vec![
        Philosopher::new("Judith Butler"),
        Philosopher::new("Gilles Deleuze"),
        Philosopher::new("Karl Marx"),
        Philosopher::new("Emma Goldman"),
        Philosopher::new("Michel Foucault"),
    ];

    let handles: Vec<_> = philosophers.into_iter().map(|p| {
        thread::spawn(move || {
            p.eat();
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }
}
```

All we’ve done is change the loop in `main()`, and added a second one! Here’s the
first change:

```rust,ignore
let handles: Vec<_> = philosophers.into_iter().map(|p| {
    thread::spawn(move || {
        p.eat();
    })
}).collect();
```

While this is only five lines, they’re a dense five. Let’s break it down.

```rust,ignore
let handles: Vec<_> =
```

We introduce a new binding, called `handles`. We’ve given it this name because
we are going to make some new threads, and that will return some handles to those
threads that let us control their operation. We need to explicitly annotate
the type here, though, due to an issue we’ll talk about later. The `_` is
a type placeholder. We’re saying “`handles` is a vector of something, but you
can figure out what that something is, Rust.”

```rust,ignore
philosophers.into_iter().map(|p| {
```

We take our list of philosophers and call `into_iter()` on it. This creates an
iterator that takes ownership of each philosopher. We need to do this to pass
them to our threads. We take that iterator and call `map` on it, which takes a
closure as an argument and calls that closure on each element in turn.

```rust,ignore
    thread::spawn(move || {
        p.eat();
    })
```

Here’s where the concurrency happens. The `thread::spawn` function takes a closure
as an argument and executes that closure in a new thread. This closure needs
an extra annotation, `move`, to indicate that the closure is going to take
ownership of the values it’s capturing. Primarily, the `p` variable of the
`map` function.

Inside the thread, all we do is call `eat()` on `p`. Also note that the call to `thread::spawn` lacks a trailing semicolon, making this an expression. This distinction is important, yielding the correct return value. For more details, read [Expressions vs. Statements][es].

[es]: functions.html#expressions-vs-statements

```rust,ignore
}).collect();
```

Finally, we take the result of all those `map` calls and collect them up.
`collect()` will make them into a collection of some kind, which is why we
needed to annotate the return type: we want a `Vec<T>`. The elements are the
return values of the `thread::spawn` calls, which are handles to those threads.
Whew!

```rust,ignore
for h in handles {
    h.join().unwrap();
}
```

At the end of `main()`, we loop through the handles and call `join()` on them,
which blocks execution until the thread has completed execution. This ensures
that the threads complete their work before the program exits.

If you run this program, you’ll see that the philosophers eat out of order!
We have multi-threading!

```text
Judith Butler is eating.
Gilles Deleuze is eating.
Karl Marx is eating.
Emma Goldman is eating.
Michel Foucault is eating.
Judith Butler is done eating.
Gilles Deleuze is done eating.
Karl Marx is done eating.
Emma Goldman is done eating.
Michel Foucault is done eating.
```

But what about the forks? We haven’t modeled them at all yet.

To do that, let’s make a new `struct`:

```rust
use std::sync::Mutex;

struct Table {
    forks: Vec<Mutex<()>>,
}
```

This `Table` has a vector of `Mutex`es. A mutex is a way to control
concurrency: only one thread can access the contents at once. This is exactly
the property we need with our forks. We use an empty tuple, `()`, inside the
mutex, since we’re not actually going to use the value, just hold onto it.

Let’s modify the program to use the `Table`:

```rust
use std::thread;
use std::time::Duration;
use std::sync::{Mutex, Arc};

struct Philosopher {
    name: String,
    left: usize,
    right: usize,
}

impl Philosopher {
    fn new(name: &str, left: usize, right: usize) -> Philosopher {
        Philosopher {
            name: name.to_string(),
            left: left,
            right: right,
        }
    }

    fn eat(&self, table: &Table) {
        let _left = table.forks[self.left].lock().unwrap();
        thread::sleep(Duration::from_millis(150));
        let _right = table.forks[self.right].lock().unwrap();

        println!("{} is eating.", self.name);

        thread::sleep(Duration::from_millis(1000));

        println!("{} is done eating.", self.name);
    }
}

struct Table {
    forks: Vec<Mutex<()>>,
}

fn main() {
    let table = Arc::new(Table { forks: vec![
        Mutex::new(()),
        Mutex::new(()),
        Mutex::new(()),
        Mutex::new(()),
        Mutex::new(()),
    ]});

    let philosophers = vec![
        Philosopher::new("Judith Butler", 0, 1),
        Philosopher::new("Gilles Deleuze", 1, 2),
        Philosopher::new("Karl Marx", 2, 3),
        Philosopher::new("Emma Goldman", 3, 4),
        Philosopher::new("Michel Foucault", 0, 4),
    ];

    let handles: Vec<_> = philosophers.into_iter().map(|p| {
        let table = table.clone();

        thread::spawn(move || {
            p.eat(&table);
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }
}
```

Lots of changes! However, with this iteration, we’ve got a working program.
Let’s go over the details:

```rust,ignore
use std::sync::{Mutex, Arc};
```

We’re going to use another structure from the `std::sync` package: `Arc<T>`.
We’ll talk more about it when we use it.

```rust,ignore
struct Philosopher {
    name: String,
    left: usize,
    right: usize,
}
```

We need to add two more fields to our `Philosopher`. Each philosopher is going
to have two forks: the one on their left, and the one on their right.
We’ll use the `usize` type to indicate them, as it’s the type that you index
vectors with. These two values will be the indexes into the `forks` our `Table`
has.

```rust,ignore
fn new(name: &str, left: usize, right: usize) -> Philosopher {
    Philosopher {
        name: name.to_string(),
        left: left,
        right: right,
    }
}
```

We now need to construct those `left` and `right` values, so we add them to
`new()`.

```rust,ignore
fn eat(&self, table: &Table) {
    let _left = table.forks[self.left].lock().unwrap();
    thread::sleep(Duration::from_millis(150));
    let _right = table.forks[self.right].lock().unwrap();

    println!("{} is eating.", self.name);

    thread::sleep(Duration::from_millis(1000));

    println!("{} is done eating.", self.name);
}
```

We have three new lines. We’ve added an argument, `table`. We access the
`Table`’s list of forks, and then use `self.left` and `self.right` to access
the fork at that particular index. That gives us access to the `Mutex` at that
index, and we call `lock()` on it. If the mutex is currently being accessed by
someone else, we’ll block until it becomes available. We have also a call to
`thread::sleep` between the moment the first fork is picked and the moment the
second forked is picked, as the process of picking up the fork is not
immediate.

The call to `lock()` might fail, and if it does, we want to crash. In this
case, the error that could happen is that the mutex is [‘poisoned’][poison],
which is what happens when the thread panics while the lock is held. Since this
shouldn’t happen, we just use `unwrap()`.

[poison]: ../std/sync/struct.Mutex.html#poisoning

One other odd thing about these lines: we’ve named the results `_left` and
`_right`. What’s up with that underscore? Well, we aren’t planning on
_using_ the value inside the lock. We just want to acquire it. As such,
Rust will warn us that we never use the value. By using the underscore,
we tell Rust that this is what we intended, and it won’t throw a warning.

What about releasing the lock? Well, that will happen when `_left` and
`_right` go out of scope, automatically.

```rust,ignore
    let table = Arc::new(Table { forks: vec![
        Mutex::new(()),
        Mutex::new(()),
        Mutex::new(()),
        Mutex::new(()),
        Mutex::new(()),
    ]});
```

Next, in `main()`, we make a new `Table` and wrap it in an `Arc<T>`.
‘arc’ stands for ‘atomic reference count’, and we need that to share
our `Table` across multiple threads. As we share it, the reference
count will go up, and when each thread ends, it will go back down.


```rust,ignore
let philosophers = vec![
    Philosopher::new("Judith Butler", 0, 1),
    Philosopher::new("Gilles Deleuze", 1, 2),
    Philosopher::new("Karl Marx", 2, 3),
    Philosopher::new("Emma Goldman", 3, 4),
    Philosopher::new("Michel Foucault", 0, 4),
];
```

We need to pass in our `left` and `right` values to the constructors for our
`Philosopher`s. But there’s one more detail here, and it’s _very_ important. If
you look at the pattern, it’s all consistent until the very end. Monsieur
Foucault should have `4, 0` as arguments, but instead, has `0, 4`. This is what
prevents deadlock, actually: one of our philosophers is left handed! This is
one way to solve the problem, and in my opinion, it’s the simplest. If you
change the order of the parameters, you will be able to observe the deadlock
taking place.

```rust,ignore
let handles: Vec<_> = philosophers.into_iter().map(|p| {
    let table = table.clone();

    thread::spawn(move || {
        p.eat(&table);
    })
}).collect();
```

Finally, inside of our `map()`/`collect()` loop, we call `table.clone()`. The
`clone()` method on `Arc<T>` is what bumps up the reference count, and when it
goes out of scope, it decrements the count. This is needed so that we know how
many references to `table` exist across our threads. If we didn’t have a count,
we wouldn’t know how to deallocate it.

You’ll notice we can introduce a new binding to `table` here, and it will
shadow the old one. This is often used so that you don’t need to come up with
two unique names.

With this, our program works! Only two philosophers can eat at any one time,
and so you’ll get some output like this:

```text
Gilles Deleuze is eating.
Emma Goldman is eating.
Emma Goldman is done eating.
Gilles Deleuze is done eating.
Judith Butler is eating.
Karl Marx is eating.
Judith Butler is done eating.
Michel Foucault is eating.
Karl Marx is done eating.
Michel Foucault is done eating.
```

Congrats! You’ve implemented a classic concurrency problem in Rust.
