% The Rust Pointer Guide

Rust's pointers are one of its more unique and compelling features. Pointers
are also one of the more confusing topics for newcomers to Rust. They can also
be confusing for people coming from other languages that support pointers, such
as C++. This guide will help you understand this important topic.

# You don't actually need pointers, use references

I have good news for you: you probably don't need to care about pointers,
especially as you're getting started. Think of it this way: Rust is a language
that emphasizes safety. Pointers, as the joke goes, are very pointy: it's easy
to accidentally stab yourself. Therefore, Rust is made in a way such that you
don't need them very often.

"But guide!" you may cry. "My co-worker wrote a function that looks like this:

~~~rust
fn succ(x: &int) -> int { *x + 1 }
~~~

So I wrote this code to try it out:

~~~rust{.ignore}
fn main() {
    let number = 5;
    let succ_number = succ(number);
    println!("{}", succ_number);
}
~~~

And now I get an error:

~~~ {.notrust}
error: mismatched types: expected `&int` but found `<generic integer #0>` (expected &-ptr but found integral variable)
~~~

What gives? It needs a pointer! Therefore I have to use pointers!"

Turns out, you don't. All you need is a reference. Try this on for size:

~~~rust
# fn succ(x: &int) -> int { *x + 1 }
fn main() {
    let number = 5;
    let succ_number = succ(&number);
    println!("{}", succ_number);
}
~~~

It's that easy! One extra little `&` there. This code will run, and print `6`.

That's all you need to know. Your co-worker could have written the function
like this:

~~~rust
fn succ(x: int) -> int { x + 1 }

fn main() {
    let number = 5;
    let succ_number = succ(number);
    println!("{}", succ_number);
}
~~~

No pointers even needed. Then again, this is a simple example. I assume that
your real-world `succ` function is more complicated, and maybe your co-worker
had a good reason for `x` to be a pointer of some kind. In that case, references
are your best friend. Don't worry about it, life is too short.

However.

Here are the use-cases for pointers. I've prefixed them with the name of the
pointer that satisfies that use-case:

1. Owned: `Box<Trait>` must be a pointer, because you don't know the size of the
object, so indirection is mandatory.

2. Owned: You need a recursive data structure. These can be infinite sized, so
indirection is mandatory.

3. Owned: A very, very, very rare situation in which you have a *huge* chunk of
data that you wish to pass to many methods. Passing a pointer will make this
more efficient. If you're coming from another language where this technique is
common, such as C++, please read "A note..." below.

4. Reference: You're writing a function, and you need a pointer, but you don't
care about its ownership. If you make the argument a reference, callers
can send in whatever kind they want.

5. Shared: You need to share data among tasks. You can achieve that via the
`Rc` and `Arc` types.

Five exceptions. That's it. Otherwise, you shouldn't need them. Be sceptical
of pointers in Rust: use them for a deliberate purpose, not just to make the
compiler happy.

## A note for those proficient in pointers

If you're coming to Rust from a language like C or C++, you may be used to
passing things by reference, or passing things by pointer. In some languages,
like Java, you can't even have objects without a pointer to them. Therefore, if
you were writing this Rust code:

~~~rust
# fn transform(p: Point) -> Point { p }
struct Point {
    x: int,
    y: int,
}

fn main() {
    let p0 = Point { x: 5, y: 10};
    let p1 = transform(p0);
    println!("{:?}", p1);
}

~~~

I think you'd implement `transform` like this:

~~~rust
# struct Point {
#     x: int,
#     y: int,
# }
# let p0 = Point { x: 5, y: 10};
fn transform(p: &Point) -> Point {
    Point { x: p.x + 1, y: p.y + 1}
}

// and change this:
let p1 = transform(&p0);
~~~

This does work, but you don't need to create those references! The better way to write this is simply:

~~~rust
struct Point {
    x: int,
    y: int,
}

fn transform(p: Point) -> Point {
    Point { x: p.x + 1, y: p.y + 1}
}

fn main() {
    let p0 = Point { x: 5, y: 10};
    let p1 = transform(p0);
    println!("{:?}", p1);
}
~~~

But won't this be inefficient? Well, that's a complicated question, but it's
important to know that Rust, like C and C++, store aggregate data types
'unboxed,' whereas languages like Java and Ruby store these types as 'boxed.'
For smaller structs, this way will be more efficient. For larger ones, it may
be less so. But don't reach for that pointer until you must! Make sure that the
struct is large enough by performing some tests before you add in the
complexity of pointers.

# Owned Pointers

Owned pointers are the conceptually simplest kind of pointer in Rust. A rough
approximation of owned pointers follows:

1. Only one owned pointer may exist to a particular place in memory. It may be
borrowed from that owner, however.

2. The Rust compiler uses static analysis to determine where the pointer is in
scope, and handles allocating and de-allocating that memory. Owned pointers are
not garbage collected.

These two properties make for three use cases.

## References to Traits

Traits must be referenced through a pointer, because the struct that implements
the trait may be a different size than a different struct that implements the
trait. Therefore, unboxed traits don't make any sense, and aren't allowed.

## Recursive Data Structures

Sometimes, you need a recursive data structure. The simplest is known as a 'cons list':

~~~rust

enum List<T> {
    Nil,
    Cons(T, Box<List<T>>),
}

fn main() {
    let list: List<int> = Cons(1, box Cons(2, box Cons(3, box Nil)));
    println!("{:?}", list);
}
~~~

This prints:

~~~ {.notrust}
Cons(1, box Cons(2, box Cons(3, box Nil)))
~~~

The inner lists _must_ be an owned pointer, because we can't know how many
elements are in the list. Without knowing the length, we don't know the size,
and therefore require the indirection that pointers offer.

## Efficiency

This should almost never be a concern, but because creating an owned pointer
boxes its value, it therefore makes referring to the value the size of the box.
This may make passing an owned pointer to a function less expensive than
passing the value itself. Don't worry yourself with this case until you've
proved that it's an issue through benchmarks.

For example, this will work:

~~~rust
struct Point {
    x: int,
    y: int,
}

fn main() {
    let a = Point { x: 10, y: 20 };
    spawn(proc() {
        println!("{}", a.x);
    });
}
~~~

This struct is tiny, so it's fine. If `Point` were large, this would be more
efficient:

~~~rust
struct Point {
    x: int,
    y: int,
}

fn main() {
    let a = box Point { x: 10, y: 20 };
    spawn(proc() {
        println!("{}", a.x);
    });
}
~~~

Now it'll be copying a pointer-sized chunk of memory rather than the whole
struct.

# References

References are the third major kind of pointer Rust supports. They are
simultaneously the simplest and the most complicated kind. Let me explain:
references are considered 'borrowed' because they claim no ownership over the
data they're pointing to. They're just borrowing it for a while. So in that
sense, they're simple: just keep whatever ownership the data already has. For
example:

~~~rust
struct Point {
    x: f32,
    y: f32,
}

fn compute_distance(p1: &Point, p2: &Point) -> f32 {
    let x_d = p1.x - p2.x;
    let y_d = p1.y - p2.y;

    (x_d * x_d + y_d * y_d).sqrt()
}

fn main() {
    let origin =    &Point { x: 0.0, y: 0.0 };
    let p1     = box Point { x: 5.0, y: 3.0 };

    println!("{:?}", compute_distance(origin, p1));
}
~~~

This prints `5.83095189`. You can see that the `compute_distance` function
takes in two references, a reference to a value on the stack, and a reference
to a value in a box.
Of course, if this were a real program, we wouldn't have any of these pointers,
they're just there to demonstrate the concepts.

So how is this hard? Well, because we're ignoring ownership, the compiler needs
to take great care to make sure that everything is safe. Despite their complete
safety, a reference's representation at runtime is the same as that of
an ordinary pointer in a C program. They introduce zero overhead. The compiler
does all safety checks at compile time.

This theory is called 'region pointers' and you can read more about it
[here](http://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf).
Region pointers evolved into what we know today as 'lifetimes'.

Here's the simple explanation: would you expect this code to compile?

~~~rust{.ignore}
fn main() {
    println!("{}", x);
    let x = 5;
}
~~~

Probably not. That's because you know that the name `x` is valid from where
it's declared to when it goes out of scope. In this case, that's the end of
the `main` function. So you know this code will cause an error. We call this
duration a 'lifetime'. Let's try a more complex example:

~~~rust
fn main() {
    let mut x = box 5;
    if *x < 10 {
        let y = &x;
        println!("Oh no: {:?}", y);
        return;
    }
    *x -= 1;
    println!("Oh no: {:?}", x);
}
~~~

Here, we're borrowing a pointer to `x` inside of the `if`. The compiler, however,
is able to determine that that pointer will go out of scope without `x` being
mutated, and therefore, lets us pass. This wouldn't work:

~~~rust{.ignore}
fn main() {
    let mut x = box 5;
    if *x < 10 {
        let y = &x;
        *x -= 1;

        println!("Oh no: {:?}", y);
        return;
    }
    *x -= 1;
    println!("Oh no: {:?}", x);
}
~~~

It gives this error:

~~~ {.notrust}
test.rs:5:8: 5:10 error: cannot assign to `*x` because it is borrowed
test.rs:5         *x -= 1;
                  ^~
test.rs:4:16: 4:18 note: borrow of `*x` occurs here
test.rs:4         let y = &x;
                          ^~
~~~

As you might guess, this kind of analysis is complex for a human, and therefore
hard for a computer, too! There is an entire [guide devoted to references
and lifetimes](guide-lifetimes.html) that goes into lifetimes in
great detail, so if you want the full details, check that out.

# Returning Pointers

We've talked a lot about functions that accept various kinds of pointers, but
what about returning them? In general, it is better to let the caller decide
how to use a function's output, instead of assuming a certain type of pointer
is best.

What does that mean? Don't do this:

~~~rust
fn foo(x: Box<int>) -> Box<int> {
    return box *x;
}

fn main() {
    let x = box 5;
    let y = foo(x);
}
~~~

Do this:

~~~rust
fn foo(x: Box<int>) -> int {
    return *x;
}

fn main() {
    let x = box 5;
    let y = box foo(x);
}
~~~

This gives you flexibility, without sacrificing performance.

You may think that this gives us terrible performance: return a value and then
immediately box it up ?! Isn't that the worst of both worlds? Rust is smarter
than that. There is no copy in this code. `main` allocates enough room for the
`box int`, passes a pointer to that memory into `foo` as `x`, and then `foo` writes
the value straight into that pointer. This writes the return value directly into
the allocated box.

This is important enough that it bears repeating: pointers are not for optimizing
returning values from your code. Allow the caller to choose how they want to
use your output.

# Related Resources

* [Lifetimes guide](guide-lifetimes.html)
