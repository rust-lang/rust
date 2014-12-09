% Error Handling in Rust

> The best-laid plans of mice and men
> Often go awry
>
> "Tae a Moose", Robert Burns

Sometimes, things just go wrong. It's important to have a plan for when the
inevitable happens. Rust has rich support for handling errors that may (let's
be honest: will) occur in your programs.

There are two main kinds of errors that can occur in your programs: failures,
and panics. Let's talk about the difference between the two, and then discuss
how to handle each. Then, we'll discuss upgrading failures to panics.

# Failure vs. Panic

Rust uses two terms to differentiate between two forms of error: failure, and
panic. A **failure** is an error that can be recovered from in some way. A
**panic** is an error that cannot be recovered from.

What do we mean by 'recover'? Well, in most cases, the possibility of an error
is expected. For example, consider the `from_str` function:

```{rust,ignore}
from_str("5");
```

This function takes a string argument and converts it into another type. But
because it's a string, you can't be sure that the conversion actually works.
For example, what should this convert to?

```{rust,ignore}
from_str("hello5world");
```

This won't work. So we know that this function will only work properly for some
inputs. It's expected behavior. We call this kind of error 'failure.'

On the other hand, sometimes, there are errors that are unexpected, or which
we cannot recover from. A classic example is an `assert!`:

```{rust,ignore}
assert!(x == 5);
```

We use `assert!` to declare that something is true. If it's not true, something
is very wrong. Wrong enough that we can't continue with things in the current
state. Another example is using the `unreachable!()` macro

```{rust,ignore}
enum Event {
    NewRelease,
}

fn probability(_: &Event) -> f64 {
    // real implementation would be more complex, of course
    0.95
}

fn descriptive_probability(event: Event) -> &'static str {
    match probability(&event) {
        1.00          => "certain",
        0.00          => "impossible",
        0.00 ... 0.25 => "very unlikely",
        0.25 ... 0.50 => "unlikely",
        0.50 ... 0.75 => "likely",
        0.75 ... 1.00  => "very likely",
    }
}

fn main() {
    std::io::println(descriptive_probability(NewRelease));
}
```

This will give us an error:

```{notrust}
error: non-exhaustive patterns: `_` not covered [E0004]
```

While we know that we've covered all possible cases, Rust can't tell. It
doesn't know that probability is between 0.0 and 1.0. So we add another case:

```rust
use Event::NewRelease;

enum Event {
    NewRelease,
}

fn probability(_: &Event) -> f64 {
    // real implementation would be more complex, of course
    0.95
}

fn descriptive_probability(event: Event) -> &'static str {
    match probability(&event) {
        1.00          => "certain",
        0.00          => "impossible",
        0.00 ... 0.25 => "very unlikely",
        0.25 ... 0.50 => "unlikely",
        0.50 ... 0.75 => "likely",
        0.75 ... 1.00  => "very likely",
        _ => unreachable!()
    }
}

fn main() {
    println!("{}", descriptive_probability(NewRelease));
}
```

We shouldn't ever hit the `_` case, so we use the `unreachable!()` macro to
indicate this. `unreachable!()` gives a different kind of error than `Result`.
Rust calls these sorts of errors 'panics.'

# Handling errors with `Option` and `Result`

The simplest way to indicate that a function may fail is to use the `Option<T>`
type. Remember our `from_str()` example? Here's its type signature:

```{rust,ignore}
pub fn from_str<A: FromStr>(s: &str) -> Option<A>
```

`from_str()` returns an `Option<A>`. If the conversion succeeds, it will return
`Some(value)`, and if it fails, it will return `None`.

This is appropriate for the simplest of cases, but doesn't give us a lot of
information in the failure case. What if we wanted to know _why_ the conversion
failed? For this, we can use the `Result<T, E>` type. It looks like this:

```rust
enum Result<T, E> {
   Ok(T),
   Err(E)
}
```

This enum is provided by Rust itself, so you don't need to define it to use it
in your code. The `Ok(T)` variant represents a success, and the `Err(E)` variant
represents a failure. Returning a `Result` instead of an `Option` is recommended
for all but the most trivial of situations.

Here's an example of using `Result`:

```rust
#[deriving(Show)]
enum Version { Version1, Version2 }

#[deriving(Show)]
enum ParseError { InvalidHeaderLength, InvalidVersion }

fn parse_version(header: &[u8]) -> Result<Version, ParseError> {
    if header.len() < 1 {
        return Err(ParseError::InvalidHeaderLength);
    }
    match header[0] {
        1 => Ok(Version::Version1),
        2 => Ok(Version::Version2),
        _ => Err(ParseError::InvalidVersion)
    }
}

let version = parse_version(&[1, 2, 3, 4]);
match version {
    Ok(v) => {
        println!("working with version: {}", v);
    }
    Err(e) => {
        println!("error parsing header: {}", e);
    }
}
```

This function makes use of an enum, `ParseError`, to enumerate the various
errors that can occur.

# Non-recoverable errors with `panic!`

In the case of an error that is unexpected and not recoverable, the `panic!`
macro will induce a panic. This will crash the current task, and give an error:

```{rust,ignore}
panic!("boom");
```

gives

```{notrust}
task '<main>' panicked at 'boom', hello.rs:2
```

when you run it.

Because these kinds of situations are relatively rare, use panics sparingly.

# Upgrading failures to panics

In certain circumstances, even though a function may fail, we may want to treat
it as a panic instead. For example, `io::stdin().read_line()` returns an
`IoResult<String>`, a form of `Result`, when there is an error reading the
line. This allows us to handle and possibly recover from this sort of error.

If we don't want to handle this error, and would rather just abort the program,
we can use the `unwrap()` method:

```{rust,ignore}
io::stdin().read_line().unwrap();
```

`unwrap()` will `panic!` if the `Option` is `None`. This basically says "Give
me the value, and if something goes wrong, just crash." This is less reliable
than matching the error and attempting to recover, but is also significantly
shorter. Sometimes, just crashing is appropriate.

There's another way of doing this that's a bit nicer than `unwrap()`:

```{rust,ignore}
let input = io::stdin().read_line()
                       .ok()
                       .expect("Failed to read line");
```
`ok()` converts the `IoResult` into an `Option`, and `expect()` does the same
thing as `unwrap()`, but takes a message. This message is passed along to the
underlying `panic!`, providing a better error message if the code errors.
