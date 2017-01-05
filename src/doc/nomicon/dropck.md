% Drop Check

We have seen how lifetimes provide us some fairly simple rules for ensuring
that we never read dangling references. However up to this point we have only ever
interacted with the *outlives* relationship in an inclusive manner. That is,
when we talked about `'a: 'b`, it was ok for `'a` to live *exactly* as long as
`'b`. At first glance, this seems to be a meaningless distinction. Nothing ever
gets dropped at the same time as another, right? This is why we used the
following desugaring of `let` statements:

```rust,ignore
let x;
let y;
```

```rust,ignore
{
    let x;
    {
        let y;
    }
}
```

Each creates its own scope, clearly establishing that one drops before the
other. However, what if we do the following?

```rust,ignore
let (x, y) = (vec![], vec![]);
```

Does either value strictly outlive the other? The answer is in fact *no*,
neither value strictly outlives the other. Of course, one of x or y will be
dropped before the other, but the actual order is not specified. Tuples aren't
special in this regard; composite structures just don't guarantee their
destruction order as of Rust 1.0.

We *could* specify this for the fields of built-in composites like tuples and
structs. However, what about something like Vec? Vec has to manually drop its
elements via pure-library code. In general, anything that implements Drop has
a chance to fiddle with its innards during its final death knell. Therefore
the compiler can't sufficiently reason about the actual destruction order
of the contents of any type that implements Drop.

So why do we care? We care because if the type system isn't careful, it could
accidentally make dangling pointers. Consider the following simple program:

```rust
struct Inspector<'a>(&'a u8);

fn main() {
    let (inspector, days);
    days = Box::new(1);
    inspector = Inspector(&days);
}
```

This program is totally sound and compiles today. The fact that `days` does
not *strictly* outlive `inspector` doesn't matter. As long as the `inspector`
is alive, so is days.

However if we add a destructor, the program will no longer compile!

```rust,ignore
struct Inspector<'a>(&'a u8);

impl<'a> Drop for Inspector<'a> {
    fn drop(&mut self) {
        println!("I was only {} days from retirement!", self.0);
    }
}

fn main() {
    let (inspector, days);
    days = Box::new(1);
    inspector = Inspector(&days);
    // Let's say `days` happens to get dropped first.
    // Then when Inspector is dropped, it will try to read free'd memory!
}
```

```text
<anon>:12:28: 12:32 error: `days` does not live long enough
<anon>:12     inspector = Inspector(&days);
                                     ^~~~
<anon>:9:11: 15:2 note: reference must be valid for the block at 9:10...
<anon>:9 fn main() {
<anon>:10     let (inspector, days);
<anon>:11     days = Box::new(1);
<anon>:12     inspector = Inspector(&days);
<anon>:13     // Let's say `days` happens to get dropped first.
<anon>:14     // Then when Inspector is dropped, it will try to read free'd memory!
          ...
<anon>:10:27: 15:2 note: ...but borrowed value is only valid for the block suffix following statement 0 at 10:26
<anon>:10     let (inspector, days);
<anon>:11     days = Box::new(1);
<anon>:12     inspector = Inspector(&days);
<anon>:13     // Let's say `days` happens to get dropped first.
<anon>:14     // Then when Inspector is dropped, it will try to read free'd memory!
<anon>:15 }
```

Implementing Drop lets the Inspector execute some arbitrary code during its
death. This means it can potentially observe that types that are supposed to
live as long as it does actually were destroyed first.

Interestingly, only generic types need to worry about this. If they aren't
generic, then the only lifetimes they can harbor are `'static`, which will truly
live *forever*. This is why this problem is referred to as *sound generic drop*.
Sound generic drop is enforced by the *drop checker*. As of this writing, some
of the finer details of how the drop checker validates types is totally up in
the air. However The Big Rule is the subtlety that we have focused on this whole
section:

**For a generic type to soundly implement drop, its generics arguments must
strictly outlive it.**

Obeying this rule is (usually) necessary to satisfy the borrow
checker; obeying it is sufficient but not necessary to be
sound. That is, if your type obeys this rule then it's definitely
sound to drop.

The reason that it is not always necessary to satisfy the above rule
is that some Drop implementations will not access borrowed data even
though their type gives them the capability for such access.

For example, this variant of the above `Inspector` example will never
access borrowed data:

```rust,ignore
struct Inspector<'a>(&'a u8, &'static str);

impl<'a> Drop for Inspector<'a> {
    fn drop(&mut self) {
        println!("Inspector(_, {}) knows when *not* to inspect.", self.1);
    }
}

fn main() {
    let (inspector, days);
    days = Box::new(1);
    inspector = Inspector(&days, "gadget");
    // Let's say `days` happens to get dropped first.
    // Even when Inspector is dropped, its destructor will not access the
    // borrowed `days`.
}
```

Likewise, this variant will also never access borrowed data:

```rust,ignore
use std::fmt;

struct Inspector<T: fmt::Display>(T, &'static str);

impl<T: fmt::Display> Drop for Inspector<T> {
    fn drop(&mut self) {
        println!("Inspector(_, {}) knows when *not* to inspect.", self.1);
    }
}

fn main() {
    let (inspector, days): (Inspector<&u8>, Box<u8>);
    days = Box::new(1);
    inspector = Inspector(&days, "gadget");
    // Let's say `days` happens to get dropped first.
    // Even when Inspector is dropped, its destructor will not access the
    // borrowed `days`.
}
```

However, *both* of the above variants are rejected by the borrow
checker during the analysis of `fn main`, saying that `days` does not
live long enough.

The reason is that the borrow checking analysis of `main` does not
know about the internals of each Inspector's Drop implementation.  As
far as the borrow checker knows while it is analyzing `main`, the body
of an inspector's destructor might access that borrowed data.

Therefore, the drop checker forces all borrowed data in a value to
strictly outlive that value.

# An Escape Hatch

The precise rules that govern drop checking may be less restrictive in
the future.

The current analysis is deliberately conservative and trivial; it forces all
borrowed data in a value to outlive that value, which is certainly sound.

Future versions of the language may make the analysis more precise, to
reduce the number of cases where sound code is rejected as unsafe.
This would help address cases such as the two Inspectors above that
know not to inspect during destruction.

In the meantime, there is an unstable attribute that one can use to
assert (unsafely) that a generic type's destructor is *guaranteed* to
not access any expired data, even if its type gives it the capability
to do so.

That attribute is called `unsafe_destructor_blind_to_params`.
To deploy it on the Inspector example from above, we would write:

```rust,ignore
struct Inspector<'a>(&'a u8, &'static str);

impl<'a> Drop for Inspector<'a> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        println!("Inspector(_, {}) knows when *not* to inspect.", self.1);
    }
}
```

This attribute has the word `unsafe` in it because the compiler is not
checking the implicit assertion that no potentially expired data
(e.g. `self.0` above) is accessed.

It is sometimes obvious that no such access can occur, like the case above.
However, when dealing with a generic type parameter, such access can
occur indirectly. Examples of such indirect access are:

 * invoking a callback,
 * via a trait method call.

(Future changes to the language, such as impl specialization, may add
other avenues for such indirect access.)

Here is an example of invoking a callback:

```rust,ignore
struct Inspector<T>(T, &'static str, Box<for <'r> fn(&'r T) -> String>);

impl<T> Drop for Inspector<T> {
    fn drop(&mut self) {
        // The `self.2` call could access a borrow e.g. if `T` is `&'a _`.
        println!("Inspector({}, {}) unwittingly inspects expired data.",
                 (self.2)(&self.0), self.1);
    }
}
```

Here is an example of a trait method call:

```rust,ignore
use std::fmt;

struct Inspector<T: fmt::Display>(T, &'static str);

impl<T: fmt::Display> Drop for Inspector<T> {
    fn drop(&mut self) {
        // There is a hidden call to `<T as Display>::fmt` below, which
        // could access a borrow e.g. if `T` is `&'a _`
        println!("Inspector({}, {}) unwittingly inspects expired data.",
                 self.0, self.1);
    }
}
```

And of course, all of these accesses could be further hidden within
some other method invoked by the destructor, rather than being written
directly within it.

In all of the above cases where the `&'a u8` is accessed in the
destructor, adding the `#[unsafe_destructor_blind_to_params]`
attribute makes the type vulnerable to misuse that the borrower
checker will not catch, inviting havoc. It is better to avoid adding
the attribute.

# Is that all about drop checker?

It turns out that when writing unsafe code, we generally don't need to
worry at all about doing the right thing for the drop checker. However there
is one special case that you need to worry about, which we will look at in
the next section.

