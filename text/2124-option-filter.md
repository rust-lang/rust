- Feature Name: option_filter
- Start Date: 2017-08-21
- RFC PR: https://github.com/rust-lang/rfcs/pull/2124
- Rust Issue: https://github.com/rust-lang/rust/issues/45860

# Summary
[summary]: #summary

Add the method `Option::filter<P>(self, predicate: P) -> Self` to the
standard library. This method makes it possible to easily throw away a `Some`
value depending on a given predicate. The call `opt.filter(p)` is equivalent
to `opt.into_iter().filter(p).next()`.

```rust
assert_eq!(Some(3).filter(|_| true)), Some(3));
assert_eq!(Some(3).filter(|_| false)), None);
assert_eq!(None.filter(|_| true), None);
```

# Motivation
[motivation]: #motivation

The `Option` type has plenty of methods, every single one intended to help the
user write short code dealing with this ubiquitous type. If we would not care
about convenience when dealing with `Option`, the type would not have nearly
as many methods.

Just like other methods, `filter()` is a useful method in *certain*
situations. While it is not nearly as important as `map()`, it is very handy
in many situations. The feedback on the [corresponding `rfcs`-issue][issue]
clearly shows that many people encountered a situation in which `filter()`
would have been helpful.

Consider this tiny example:

```rust
let api_key = std::env::arg("APIKEY").ok()
    .filter(|key| key.starts_with("api"));
```

Here is another example showing tree traversal with a queue:

```rust
let mut queue = VecDeque::new();
queue.push_back(tree.root());

// We want to visit all nodes in breadth first search order, but stop
// immediately once we found a leaf node.
while let Some(node) = queue.pop_front().filter(|node| !node.is_leaf()) {
    queue.extend(node.children());
}
```

Additionally, adding `filter()` would make the interfaces of `Option` and
`Iterator` more consistent. Both types already shared a handful of methods
with identical names and functions, most importantly `map()`. Adding another
such method would make the whole interface feel more consistent.

In the following example the programmer can easily swap `nth()` and `filter()`
statements, if they decide they want to allow the `-j` parameter at any
position.

```rust
let num_threads = std::env::args()
    .nth(1)
    .filter(|arg| arg.starts_with("-j"))
    .and_then(|arg| arg[2..].parse().ok());

```

`filter()` can be especially useful for integration into existing method-
chains. Here is a slightly more complicated example which is taken from an
existing, real web app's session management. Note that each line introduces a
new reason to reject the session.

```rust
// Check if there is a session-cookie
let session = cookies.get(SESSION_COOKIE_NAME)
    // Try to decode the cookie's value as hexadecimal string
    .and_then(|cookie| hex::decode(cookie.value()).ok())
    // Make sure the session id has the correct length
    .filter(|session_id| session_id.len() == SESSION_ID_LEN)
    // Try to find the session with the given ID in the database
    .and_then(|session_id| db.find_session_by_id(session_id));
```

All these examples would be less easy to read without `filter()`. There are
two main ways to achieve something equivalent to `filter(p)`:

- `opt.into_iter().filter(p).next()`: notably longer and the `next()` feels
  semantically wrong.
- `opt.and_then(|v| if p(&v) { Some(v) } else { None })`: notably longer and a
  questionable single-line `if-else`.


[issue]: https://github.com/rust-lang/rfcs/issues/1485

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

A possible documentation of the method:

> ```rust
> fn filter<P>(self, predicate: P) -> Self
>     where P: FnOnce(&T) -> bool
> ```
>
> Returns `None` if the option is `None`, otherwise calls `predicate` with the
> wrapped value and returns:
>
> - `Some(t)` if `predicate` returns `true` (where `t` is the wrapped value),
>    and
> - `None` if `predicate` returns `false`.
>
> This function works similar to `Iterator::filter()`. You can imagine the
> `Option<T>` being an iterator over one or zero elements. `filter()` lets
> you decide which elements to keep.
>
> # Examples
>
> ```rust
> fn is_even(n: i32) -> bool {
>     n % 2 == 0
> }
>
> assert_eq!(None.filter(is_even), None);
> assert_eq!(Some(3).filter(is_even), None);
> assert_eq!(Some(4).filter(is_even), Some(4));
> ```
>

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

It is hopefully sufficiently clear how `filter()` is supposed to work from the
explanations above. Here is one example implementation:

```rust
impl<T> Option<T> {
    pub fn filter<P>(self, predicate: P) -> Self
        where P: FnOnce(&T) -> bool
    {
        match self {
            Some(x) => {
                if predicate(&x) {
                    Some(x)
                } else {
                    None
                }
            }
            None => None,
        }
    }
}
```

# Drawbacks
[drawbacks]: #drawbacks

It increases the size of the standard library by a tiny bit.

# Rationale and Alternatives
[alternatives]: #alternatives

- Don't do anything.

# Unresolved questions
[unresolved]: #unresolved-questions

### Maybe `filter()` wouldn't be used a lot.

The feature proposed in this RFC is already implemented in the
[`option-filter` crate][crate]. This crate hasn't been used a lot (only
around 1500 downloads at the time of writing this). Thus, it makes sense to ask whether people would actually use the `filter()` method. However, there
are many other reasons for not using this crate:

- The programmer doesn't know about the crate
- The programmer knows about the crate, but doesn't want to have too many tiny
  dependencies in their project
- The programmer knows about the crate, but they decided it's too much work to
  use the crate.

  A simple calculation: using the crate would require around 80 new characters
  (`option-filter = "*"` + `extern crate option_filter;` +
  `use option_filter::OptionFilterExt;`) in at least 2, probably 3, files. On
  the other hand, using the `.and_then()` workaround shown above would only
  need 39 more characters than `filter()` and wouldn't require opening other
  files.

According to the assessment of this RFC's author, the mentioned crate is not
used for reasons independently of `filter()`'s usefulness.

Reading the comments and looking at the feedback in [this thread][rfcs-issue],
it's clear that there are at least some people openly requesting this feature.
And to give a specific example: this RFC's author wanted to use `filter()` a
whole lot more often than he used some of the other methods of `Option` (like
`map_or_else()` and `ok_or_else()`).


[crate]: https://crates.io/crates/option-filter
[rfcs-issue]: https://github.com/rust-lang/rfcs/issues/1485
