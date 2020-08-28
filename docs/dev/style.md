Our approach to "clean code" is two-fold:

* We generally don't block PRs on style changes.
* At the same time, all code in rust-analyzer is constantly refactored.

It is explicitly OK for a reviewer to flag only some nits in the PR, and then send a follow-up cleanup PR for things which are easier to explain by example, cc-ing the original author.
Sending small cleanup PRs (like renaming a single local variable) is encouraged.

# Scale of Changes

Everyone knows that it's better to send small & focused pull requests.
The problem is, sometimes you *have* to, eg, rewrite the whole compiler, and that just doesn't fit into a set of isolated PRs.

The main things to keep an eye on are the boundaries between various components.
There are three kinds of changes:

1. Internals of a single component are changed.
   Specifically, you don't change any `pub` items.
   A good example here would be an addition of a new assist.

2. API of a component is expanded.
   Specifically, you add a new `pub` function which wasn't there before.
   A good example here would be expansion of assist API, for example, to implement lazy assists or assists groups.

3. A new dependency between components is introduced.
   Specifically, you add a `pub use` reexport from another crate or you add a new line to the `[dependencies]` section of `Cargo.toml`.
   A good example here would be adding reference search capability to the assists crates.

For the first group, the change is generally merged as long as:

* it works for the happy case,
* it has tests,
* it doesn't panic for the unhappy case.

For the second group, the change would be subjected to quite a bit of scrutiny and iteration.
The new API needs to be right (or at least easy to change later).
The actual implementation doesn't matter that much.
It's very important to minimize the amount of changed lines of code for changes of the second kind.
Often, you start doing a change of the first kind, only to realise that you need to elevate to a change of the second kind.
In this case, we'll probably ask you to split API changes into a separate PR.

Changes of the third group should be pretty rare, so we don't specify any specific process for them.
That said, adding an innocent-looking `pub use` is a very simple way to break encapsulation, keep an eye on it!

Note: if you enjoyed this abstract hand-waving about boundaries, you might appreciate
https://www.tedinski.com/2018/02/06/system-boundaries.html

# Crates.io Dependencies

We try to be very conservative with usage of crates.io dependencies.
Don't use small "helper" crates (exception: `itertools` is allowed).
If there's some general reusable bit of code you need, consider adding it to the `stdx` crate.

# Minimal Tests

Most tests in rust-analyzer start with a snippet of Rust code.
This snippets should be minimal -- if you copy-paste a snippet of real code into the tests, make sure to remove everything which could be removed.
There are many benefits to this:

* less to read or to scroll past
* easier to understand what exactly is tested
* less stuff printed during printf-debugging
* less time to run test

It also makes sense to format snippets more compactly (for example, by placing enum definitions like `enum E { Foo, Bar }` on a single line),
as long as they are still readable.

# Order of Imports

Separate import groups with blank lines.
Use one `use` per crate.

```rust
mod x;
mod y;

// First std.
use std::{ ... }

// Second, external crates (both crates.io crates and other rust-analyzer crates).
use crate_foo::{ ... }
use crate_bar::{ ... }

// Then current crate.
use crate::{}

// Finally, parent and child modules, but prefer `use crate::`.
use super::{}
```

Module declarations come before the imports.
Order them in "suggested reading order" for a person new to the code base.

# Import Style

Qualify items from `hir` and `ast`.

```rust
// Good
use syntax::ast;

fn frobnicate(func: hir::Function, strukt: ast::StructDef) {}

// Not as good
use hir::Function;
use syntax::ast::StructDef;

fn frobnicate(func: Function, strukt: StructDef) {}
```

Avoid local `use MyEnum::*` imports.

Prefer `use crate::foo::bar` to `use super::bar`.

When implementing `Debug` or `Display`, import `std::fmt`:

```rust
// Good
use std::fmt;

impl fmt::Display for RenameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { .. }
}

// Not as good
impl std::fmt::Display for RenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { .. }
}
```

# Order of Items

Optimize for the reader who sees the file for the first time, and wants to get a general idea about what's going on.
People read things from top to bottom, so place most important things first.

Specifically, if all items except one are private, always put the non-private item on top.

Put `struct`s and `enum`s first, functions and impls last.

Do

```rust
// Good
struct Foo {
    bars: Vec<Bar>
}

struct Bar;
```

rather than

```rust
// Not as good
struct Bar;

struct Foo {
    bars: Vec<Bar>
}
```

# Variable Naming

Use boring and long names for local variables ([yay code completion](https://github.com/rust-analyzer/rust-analyzer/pull/4162#discussion_r417130973)).
The default name is a lowercased name of the type: `global_state: GlobalState`.
Avoid ad-hoc acronyms and contractions, but use the ones that exist consistently (`db`, `ctx`, `acc`).

Default names:

* `res` -- "result of the function" local variable
* `it` -- I don't really care about the name
* `n_foo` -- number of foos
* `foo_idx` -- index of `foo`

# Collection types

Prefer `rustc_hash::FxHashMap` and `rustc_hash::FxHashSet` instead of the ones in `std::collections`.
They use a hasher that's slightly faster and using them consistently will reduce code size by some small amount.

# Preconditions

Express function preconditions in types and force the caller to provide them (rather than checking in callee):

```rust
// Good
fn frbonicate(walrus: Walrus) {
    ...
}

// Not as good
fn frobnicate(walrus: Option<Walrus>) {
    let walrus = match walrus {
        Some(it) => it,
        None => return,
    };
    ...
}
```

# Early Returns

Do use early returns

```rust
// Good
fn foo() -> Option<Bar> {
    if !condition() {
        return None;
    }

    Some(...)
}

// Not as good
fn foo() -> Option<Bar> {
    if condition() {
        Some(...)
    } else {
        None
    }
}
```

# Getters & Setters

If a field can have any value without breaking invariants, make the field public.
Conversely, if there is an invariant, document it, enforce it in the "constructor" function, make the field private, and provide a getter.
Never provide setters.

Getters should return borrowed data:

```rust
struct Person {
    // Invariant: never empty
    first_name: String,
    middle_name: Option<String>
}

// Good
impl Person {
    fn first_name(&self) -> &str { self.first_name.as_str() }
    fn middle_name(&self) -> Option<&str> { self.middle_name.as_ref() }
}

// Not as good
impl Person {
    fn first_name(&self) -> String { self.first_name.clone() }
    fn middle_name(&self) -> &Option<String> { &self.middle_name }
}
```


# Premature Pessimization

Avoid writing code which is slower than it needs to be.
Don't allocate a `Vec` where an iterator would do, don't allocate strings needlessly.

```rust
// Good
use itertools::Itertools;

let (first_word, second_word) = match text.split_ascii_whitespace().collect_tuple() {
    Some(it) => it,
    None => return,
}

// Not as good
let words = text.split_ascii_whitespace().collect::<Vec<_>>();
if words.len() != 2 {
    return
}
```

# Avoid Monomorphization

Rust uses monomorphization to compile generic code, meaning that for each instantiation of a generic functions with concrete types, the function is compiled afresh, *per crate*.
This allows for exceptionally good performance, but leads to increased compile times.
Runtime performance obeys 80%/20% rule -- only a small fraction of code is hot.
Compile time **does not** obey this rule -- all code has to be compiled.
For this reason, avoid making a lot of code type parametric, *especially* on the boundaries between crates.

```rust
// Good
fn frbonicate(f: impl FnMut()) {
    frobnicate_impl(&mut f)
}
fn frobnicate_impl(f: &mut dyn FnMut()) {
    // lots of code
}

// Not as good
fn frbonicate(f: impl FnMut()) {
    // lots of code
}
```

Avoid `AsRef` polymorphism, it pays back only for widely used libraries:

```rust
// Good
fn frbonicate(f: &Path) {
}

// Not as good
fn frbonicate(f: impl AsRef<Path>) {
}
```

# Documentation

For `.md` and `.adoc` files, prefer a sentence-per-line format, don't wrap lines.
If the line is too long, you want to split the sentence in two :-)

# Commit Style

We don't have specific rules around git history hygiene.
Maintaining clean git history is strongly encouraged, but not enforced.
Use rebase workflow, it's OK to rewrite history during PR review process.
After you are happy with the state of the code, please use [interactive rebase](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) to squash fixup commits.

Avoid @mentioning people in commit messages and pull request descriptions(they are added to commit message by bors).
Such messages create a lot of duplicate notification traffic during rebases.
