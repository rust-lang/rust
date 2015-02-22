% Closures

So far, we've made lots of functions in Rust, but we've given them all names.
Rust also allows us to create anonymous functions. Rust's anonymous
functions are called *closures*. By themselves, closures aren't all that
interesting, but when you combine them with functions that take closures as
arguments, really powerful things are possible.

## Declaring closures

Here's a very simple function, and its closure version:

```rust
fn add_one(x: i32) -> i32 { x + 1 }

let add_one = |x| { x + 1 };
```

Here, let's change the whitespace a little, so we can see the parts line up:

```rust
fn  add_one   (x: i32) -> i32 { x + 1 }
let add_one = |x|             { x + 1 };
```

The biggest thing you'll notice is that closures infer the types of their
arguments and return values. Second, arguments go between a set of pipes (`|`),
rather than in parentheses. Finally, to give a closure a name, we need to use
`let` to assign it to a binding.

Calling a closure looks just like calling a function. Let's change the closure
version to be bound to `plus_one` so we can compare:

```rust
fn add_one(x: i32) -> i32 { x + 1 }

let plus_one = |x| { x + 1 };

assert_eq!(6, add_one(5));
assert_eq!(6, plus_one(5));
```

So what's the big deal? Closures are useful when you want to pass a closure to
another function. But before we can learn about that, we need to talk about the
traits that closures are built on.

## `Fn`, `FnMut`, `FnOnce`

Rust's closures have a secret: they're actually just sugar for traits. In some
sense, `()` is an overloadable operator. Like other operators, you can overload
`()` by implementing a trait. In the case of `()`, it's actually three different
traits:

```rust
trait Fn<Args> {
    type Output;

    fn call(&self, args: Args) -> Self::Output;
}

trait FnMut<Args> {
    type Output;

    fn call_mut(&mut self, args: Args) -> Self::Output;
}

trait FnOnce<Args> {
    type Output;

    fn call_once(self, args: Args) -> Self::Output;
}
```

These three traits correspond to the three kinds of methods: `Fn` is `&self`,
`FnMut` is `&mut self`, and `FnOnce` is `self`. We'll talk more about these three
kinds in a moment, so for now, let's just consider `Fn`.

## Taking closures as arguments

You can declare a closure as an argument to a function like this:

```rust
fn call<F>(s: String, f: F) -> u32 
    where F: Fn(String) -> u32
{
    f(s)
}
```

This function, `call`, takes a `String` and a function from `String -> u32`,
and calls that function on the string.

Let's call `call`:

```rust
# fn call<F>(s: String, f: F) -> u32 
#    where F: Fn(String) -> u32
# { f(s) }

let forty_two = "42".to_string();
let function = |s: String| { s.parse().unwrap() };

call(forty_two, function);

// or inline
let forty_two = "42".to_string();

call(forty_two, |s| {
    s.parse().unwrap()
});
```

As you can see, when we define `function`, we need to let Rust know what
type the argument `s` is. We can do that with a colon, just like in named
`fn` declarations.

This function doesn't refer to anything but its arguments, and so doesn't
really close over anything. Let's change that:

```rust
# fn call<F>(s: String, f: F) -> u32 
#    where F: Fn(String) -> u32
# { f(s) }

let forty_two = "42".to_string();
let number = 5;

call(forty_two, |s| {
    s.parse().unwrap() + number
});
```

The Rust compiler turns this closure into something like this:

```ignore
struct ClosureEnvironment {
    number: u32
}

impl Fn(String) -> u32 for ClosureEnvironment {
    fn call(&self, (s,): (String,)) -> u32 {
	s.parse().unwrap() + self.number
    }
}

let forty_two = "42".to_string();
let number = 5;
let closure = ClosureEnvironment{ number: number };

call(forty_two, closure);
```

As you can see, we generate a new `struct` for the environment, and then implements
the `Fn` trait for that struct. Because the struct implements the trait, it can be
passed to the function, which takes something that implements that trait.

This also further explains why we need the three traits: If we only borrow our
environment, we use `Fn`, if we borrow it mutably, we use `FnMut`, and if we consume
our environment, we use `FnOnce`.

## `move` closures


## Returning closures from functions
