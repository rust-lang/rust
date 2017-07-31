- Feature Name: refcell-replace
- Start Date: 2017-06-09
- RFC PR: [rust-lang/rfcs#2057](https://github.com/rust-lang/rfcs/pull/2057)
- Rust Issue: [rust-lang/rust#43570](https://github.com/rust-lang/rust/issues/43570)

# Summary
[summary]: #summary

Add dedicated methods to RefCell for replacing and swapping the contents.
These functions will panic if the RefCell is currently borrowed,
but will otherwise behave exactly like their cousins on Cell.

# Motivation
[motivation]: #motivation

The main problem this intends to solve is that doing a replace by hand
looks like this:

```rust
let old_version = replace(&mut *some_refcell.borrow_mut(), new_version);
```

One of the most important parts of the ergonomics initiative has been reducing
"type tetris" exactly like that `&mut *`.

It also seems weird that this use-case is so much cleaner with a plain `Cell`,
even though plain `Cell` is strictly a less powerful abstraction.
Usually, people explain `RefCell` as being a superset of `Cell`,
but `RefCell` doesn't actually offer all of the functionality as seamlessly as `Cell`.

# Detailed design
[design]: #detailed-design

```rust
impl<T> RefCell<T> {
  pub fn replace(&self, t: T) -> T {
      mem::replace(&mut *self.borrow_mut(), t)
  }
  pub fn swap(&self, other: &Self) {
      mem::swap(&mut *self.borrow_mut(), &mut *other.borrow_mut())
  }
}
```

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The nicest aspect of this is that it maintains this story behind `Cell` and `RefCell`:

> `RefCell` supports everything that `Cell` does. However, it has runtime overhead,
> and it can panic.

# Drawbacks
[drawbacks]: #drawbacks

Depending on how we want people to use RefCell,
this RFC might be removing deliberate syntactic vinegar.
For example, if RefCell is used to protect a counter:

```rust
let counter_ref = counter.borrow_mut();
*counter_ref += 1;
do_some_work();
*counter_ref -= 1;
```

In this case, if `do_some_work()` tries to modify `counter`, it will panic.
Since Rust tends to value explicitness over implicitness exactly because it can surface bugs,
this code is conceptually more dangerous:

```rust
counter.replace(counter.replace(0) + 1);
do_some_work();
counter.replace(counter.replace(0) - 1);
```

Also, we're adding more specific functions to a core type.
That comes with cost in documentation and maintainance.

# Alternatives
[alternatives]: #alternatives

Besides just-write-the-reborrow,
these functions can also be put in a separate crate
with an extension trait.
This has all the disadvantages that two-line libraries usually have:

  * They tend to have low discoverability.
  * They put strain on auditing.
  * The hassle of adding an import and a toml line is as high as the reborrow.

The other alternative, as far as getting rid of the reborrow goes,
is to change the language so that it implicitly does the reborrow.
That alternative is *massively* more general,
but it also has knock-on effects throughout the rest of the language.
It also still doesn't do anything about the asymetry between Cell and RefCell.

# Unresolved questions
[unresolved]: #unresolved-questions

Should we add `RefCell::get()` and `RefCell::set()`?
The equivalent versions with borrow(mut) and clone aren't as noisy,
since all the reborrowing is done implicitly because clone is a method,
but that would bring us all the way to RefCell-as-a-Cell-superset.
