% Drop Check

We have seen how lifetimes provide us some fairly simple rules for ensuring
that never read dangling references. However up to this point we have only ever
interacted with the *outlives* relationship in an inclusive manner. That is,
when we talked about `'a: 'b`, it was ok for `'a` to live *exactly* as long as
`'b`. At first glance, this seems to be a meaningless distinction. Nothing ever
gets dropped at the same time as another, right? This is why we used the
following desugarring of `let` statements:

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
neither value  strictly outlives the other. Of course, one of x or y will be
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

Implementing Drop lets the Inspector execute some arbitrary code *during* its
death. This means it can potentially observe that types that are supposed to
live as long as it does actually were destroyed first.

Interestingly, only *generic* types need to worry about this. If they aren't
generic, then the only lifetimes they can harbor are `'static`, which will truly
live *forever*. This is why this problem is referred to as *sound generic drop*.
Sound generic drop is enforced by the *drop checker*. As of this writing, some
of the finer details of how the drop checker validates types is totally up in
the air. However The Big Rule is the subtlety that we have focused on this whole
section:

**For a generic type to soundly implement drop, its generics arguments must
strictly outlive it.**

This rule is sufficient but not necessary to satisfy the drop checker. That is,
if your type obeys this rule then it's *definitely* sound to drop. However
there are special cases where you can fail to satisfy this, but still
successfully pass the borrow checker. These are the precise rules that are
currently up in the air.

It turns out that when writing unsafe code, we generally don't need to
worry at all about doing the right thing for the drop checker. However there
is *one* special case that you need to worry about, which we will look at in
the next section.
