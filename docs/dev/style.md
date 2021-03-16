Our approach to "clean code" is two-fold:

* We generally don't block PRs on style changes.
* At the same time, all code in rust-analyzer is constantly refactored.

It is explicitly OK for a reviewer to flag only some nits in the PR, and then send a follow-up cleanup PR for things which are easier to explain by example, cc-ing the original author.
Sending small cleanup PRs (like renaming a single local variable) is encouraged.

When reviewing pull requests prefer extending this document to leaving
non-reusable comments on the pull request itself.

# General

## Scale of Changes

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
Often, you start doing a change of the first kind, only to realize that you need to elevate to a change of the second kind.
In this case, we'll probably ask you to split API changes into a separate PR.

Changes of the third group should be pretty rare, so we don't specify any specific process for them.
That said, adding an innocent-looking `pub use` is a very simple way to break encapsulation, keep an eye on it!

Note: if you enjoyed this abstract hand-waving about boundaries, you might appreciate
https://www.tedinski.com/2018/02/06/system-boundaries.html

## Crates.io Dependencies

We try to be very conservative with usage of crates.io dependencies.
Don't use small "helper" crates (exception: `itertools` is allowed).
If there's some general reusable bit of code you need, consider adding it to the `stdx` crate.

**Rationale:** keep compile times low, create ecosystem pressure for faster
compiles, reduce the number of things which might break.

## Commit Style

We don't have specific rules around git history hygiene.
Maintaining clean git history is strongly encouraged, but not enforced.
Use rebase workflow, it's OK to rewrite history during PR review process.
After you are happy with the state of the code, please use [interactive rebase](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) to squash fixup commits.

Avoid @mentioning people in commit messages and pull request descriptions(they are added to commit message by bors).
Such messages create a lot of duplicate notification traffic during rebases.

If possible, write commit messages from user's perspective:

```
# GOOD
Goto definition works inside macros

# BAD
Use original span for FileId
```

This makes it easier to prepare a changelog.

If the change adds a new user-visible functionality, consider recording a GIF with [peek](https://github.com/phw/peek) and pasting it into the PR description.

**Rationale:** clean history is potentially useful, but rarely used.
But many users read changelogs.

## Clippy

We don't enforce Clippy.
A number of default lints have high false positive rate.
Selectively patching false-positives with `allow(clippy)` is considered worse than not using Clippy at all.
There's a `cargo lint` command which runs a subset of low-FPR lints.
Careful tweaking of `lint` is welcome.
Of course, applying Clippy suggestions is welcome as long as they indeed improve the code.

**Rationale:** see [rust-lang/clippy#5537](https://github.com/rust-lang/rust-clippy/issues/5537).

# Code

## Minimal Tests

Most tests in rust-analyzer start with a snippet of Rust code.
These snippets should be minimal -- if you copy-paste a snippet of real code into the tests, make sure to remove everything which could be removed.

It also makes sense to format snippets more compactly (for example, by placing enum definitions like `enum E { Foo, Bar }` on a single line),
as long as they are still readable.

When using multiline fixtures, use unindented raw string literals:

```rust
    #[test]
    fn inline_field_shorthand() {
        check_assist(
            inline_local_variable,
            r#"
struct S { foo: i32}
fn main() {
    let $0foo = 92;
    S { foo }
}
"#,
            r#"
struct S { foo: i32}
fn main() {
    S { foo: 92 }
}
"#,
        );
    }
```

**Rationale:**

There are many benefits to this:

* less to read or to scroll past
* easier to understand what exactly is tested
* less stuff printed during printf-debugging
* less time to run test

Formatting ensures that you can use your editor's "number of selected characters" feature to correlate offsets with test's source code.

## Marked Tests

Use
[`cov_mark::hit! / cov_mark::check!`](https://github.com/matklad/cov-mark)
when testing specific conditions.
Do not place several marks into a single test or condition.
Do not reuse marks between several tests.

**Rationale:** marks provide an easy way to find the canonical test for each bit of code.
This makes it much easier to understand.

## Function Preconditions

Express function preconditions in types and force the caller to provide them (rather than checking in callee):

```rust
// GOOD
fn frobnicate(walrus: Walrus) {
    ...
}

// BAD
fn frobnicate(walrus: Option<Walrus>) {
    let walrus = match walrus {
        Some(it) => it,
        None => return,
    };
    ...
}
```

**Rationale:** this makes control flow explicit at the call site.
Call-site has more context, it often happens that the precondition falls out naturally or can be bubbled up higher in the stack.

Avoid splitting precondition check and precondition use across functions:

```rust
// GOOD
fn main() {
    let s: &str = ...;
    if let Some(contents) = string_literal_contents(s) {

    }
}

fn string_literal_contents(s: &str) -> Option<&str> {
    if s.starts_with('"') && s.ends_with('"') {
        Some(&s[1..s.len() - 1])
    } else {
        None
    }
}

// BAD
fn main() {
    let s: &str = ...;
    if is_string_literal(s) {
        let contents = &s[1..s.len() - 1];
    }
}

fn is_string_literal(s: &str) -> bool {
    s.starts_with('"') && s.ends_with('"')
}
```

In the "Not as good" version, the precondition that `1` is a valid char boundary is checked in `is_string_literal` and used in `foo`.
In the "Good" version, the precondition check and usage are checked in the same block, and then encoded in the types.

**Rationale:** non-local code properties degrade under change.

When checking a boolean precondition, prefer `if !invariant` to `if negated_invariant`:

```rust
// GOOD
if !(idx < len) {
    return None;
}

// BAD
if idx >= len {
    return None;
}
```

**Rationale:** it's useful to see the invariant relied upon by the rest of the function clearly spelled out.

## Assertions

Assert liberally.
Prefer `stdx::never!` to standard `assert!`.

## Getters & Setters

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

// GOOD
impl Person {
    fn first_name(&self) -> &str { self.first_name.as_str() }
    fn middle_name(&self) -> Option<&str> { self.middle_name.as_ref() }
}

// BAD
impl Person {
    fn first_name(&self) -> String { self.first_name.clone() }
    fn middle_name(&self) -> &Option<String> { &self.middle_name }
}
```

**Rationale:** we don't provide public API, it's cheaper to refactor than to pay getters rent.
Non-local code properties degrade under change, privacy makes invariant local.
Borrowed own data discloses irrelevant details about origin of data.
Irrelevant (neither right nor wrong) things obscure correctness.

## Useless Types

More generally, always prefer types on the left

```rust
// GOOD      BAD
&[T]         &Vec<T>
&str         &String
Option<&T>   &Option<T>
```

**Rationale:** types on the left are strictly more general.
Even when generality is not required, consistency is important.

## Constructors

Prefer `Default` to zero-argument `new` function

```rust
// GOOD
#[derive(Default)]
struct Foo {
    bar: Option<Bar>
}

// BAD
struct Foo {
    bar: Option<Bar>
}

impl Foo {
    fn new() -> Foo {
        Foo { bar: None }
    }
}
```

Prefer `Default` even it has to be implemented manually.

**Rationale:** less typing in the common case, uniformity.

Use `Vec::new` rather than `vec![]`.

**Rationale:** uniformity, strength reduction.

## Functions Over Objects

Avoid creating "doer" objects.
That is, objects which are created only to execute a single action.

```rust
// GOOD
do_thing(arg1, arg2);

// BAD
ThingDoer::new(arg1, arg2).do();
```

Note that this concerns only outward API.
When implementing `do_thing`, it might be very useful to create a context object.

```rust
pub fn do_thing(arg1: Arg1, arg2: Arg2) -> Res {
    let mut ctx = Ctx { arg1, arg2 }
    ctx.run()
}

struct Ctx {
    arg1: Arg1, arg2: Arg2
}

impl Ctx {
    fn run(self) -> Res {
        ...
    }
}
```

The difference is that `Ctx` is an impl detail here.

Sometimes a middle ground is acceptable if this can save some busywork:

```rust
ThingDoer::do(arg1, arg2);

pub struct ThingDoer {
    arg1: Arg1, arg2: Arg2,
}

impl ThingDoer {
    pub fn do(arg1: Arg1, arg2: Arg2) -> Res {
        ThingDoer { arg1, arg2 }.run()
    }
    fn run(self) -> Res {
        ...
    }
}
```

**Rationale:** not bothering the caller with irrelevant details, not mixing user API with implementor API.

## Functions with many parameters

Avoid creating functions with many optional or boolean parameters.
Introduce a `Config` struct instead.

```rust
// GOOD
pub struct AnnotationConfig {
    pub binary_target: bool,
    pub annotate_runnables: bool,
    pub annotate_impls: bool,
}

pub fn annotations(
    db: &RootDatabase,
    file_id: FileId,
    config: AnnotationConfig
) -> Vec<Annotation> {
    ...
}

// BAD
pub fn annotations(
    db: &RootDatabase,
    file_id: FileId,
    binary_target: bool,
    annotate_runnables: bool,
    annotate_impls: bool,
) -> Vec<Annotation> {
    ...
}
```

**Rationale:** reducing churn.
If the function has many parameters, they most likely change frequently.
By packing them into a struct we protect all intermediary functions from changes.

Do not implement `Default` for the `Config` struct, the caller has more context to determine better defaults.
Do not store `Config` as a part of the `state`, pass it explicitly.
This gives more flexibility for the caller.

If there is variation not only in the input parameters, but in the return type as well, consider introducing a `Command` type.

```rust
// MAYBE GOOD
pub struct Query {
    pub name: String,
    pub case_sensitive: bool,
}

impl Query {
    pub fn all(self) -> Vec<Item> { ... }
    pub fn first(self) -> Option<Item> { ... }
}

// MAYBE BAD
fn query_all(name: String, case_sensitive: bool) -> Vec<Item> { ... }
fn query_first(name: String, case_sensitive: bool) -> Option<Item> { ... }
```

## Avoid Monomorphization

Avoid making a lot of code type parametric, *especially* on the boundaries between crates.

```rust
// GOOD
fn frobnicate(f: impl FnMut()) {
    frobnicate_impl(&mut f)
}
fn frobnicate_impl(f: &mut dyn FnMut()) {
    // lots of code
}

// BAD
fn frobnicate(f: impl FnMut()) {
    // lots of code
}
```

Avoid `AsRef` polymorphism, it pays back only for widely used libraries:

```rust
// GOOD
fn frobnicate(f: &Path) {
}

// BAD
fn frobnicate(f: impl AsRef<Path>) {
}
```

**Rationale:** Rust uses monomorphization to compile generic code, meaning that for each instantiation of a generic functions with concrete types, the function is compiled afresh, *per crate*.
This allows for exceptionally good performance, but leads to increased compile times.
Runtime performance obeys 80%/20% rule -- only a small fraction of code is hot.
Compile time **does not** obey this rule -- all code has to be compiled.

## Appropriate String Types

When interfacing with OS APIs, use `OsString`, even if the original source of data is utf-8 encoded.
**Rationale:** cleanly delineates the boundary when the data goes into the OS-land.

Use `AbsPathBuf` and `AbsPath` over `std::Path`.
**Rationale:** rust-analyzer is a long-lived process which handles several projects at the same time.
It is important not to leak cwd by accident.

# Premature Pessimization

## Avoid Allocations

Avoid writing code which is slower than it needs to be.
Don't allocate a `Vec` where an iterator would do, don't allocate strings needlessly.

```rust
// GOOD
use itertools::Itertools;

let (first_word, second_word) = match text.split_ascii_whitespace().collect_tuple() {
    Some(it) => it,
    None => return,
}

// BAD
let words = text.split_ascii_whitespace().collect::<Vec<_>>();
if words.len() != 2 {
    return
}
```

**Rationale:** not allocating is almost often faster.

## Push Allocations to the Call Site

If allocation is inevitable, let the caller allocate the resource:

```rust
// GOOD
fn frobnicate(s: String) {
    ...
}

// BAD
fn frobnicate(s: &str) {
    let s = s.to_string();
    ...
}
```

**Rationale:** reveals the costs.
It is also more efficient when the caller already owns the allocation.

## Collection Types

Prefer `rustc_hash::FxHashMap` and `rustc_hash::FxHashSet` instead of the ones in `std::collections`.

**Rationale:** they use a hasher that's significantly faster and using them consistently will reduce code size by some small amount.

## Avoid Intermediate Collections

When writing a recursive function to compute a sets of things, use an accumulator parameter instead of returning a fresh collection.
Accumulator goes first in the list of arguments.

```rust
// GOOD
pub fn reachable_nodes(node: Node) -> FxHashSet<Node> {
    let mut res = FxHashSet::default();
    go(&mut res, node);
    res
}
fn go(acc: &mut FxHashSet<Node>, node: Node) {
    acc.insert(node);
    for n in node.neighbors() {
        go(acc, n);
    }
}

// BAD
pub fn reachable_nodes(node: Node) -> FxHashSet<Node> {
    let mut res = FxHashSet::default();
    res.insert(node);
    for n in node.neighbors() {
        res.extend(reachable_nodes(n));
    }
    res
}
```

**Rationale:** re-use allocations, accumulator style is more concise for complex cases.

# Style

## Order of Imports

Separate import groups with blank lines.
Use one `use` per crate.

Module declarations come before the imports.
Order them in "suggested reading order" for a person new to the code base.

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

**Rationale:** consistency.
Reading order is important for new contributors.
Grouping by crate allows to spot unwanted dependencies easier.

## Import Style

Qualify items from `hir` and `ast`.

```rust
// GOOD
use syntax::ast;

fn frobnicate(func: hir::Function, strukt: ast::Struct) {}

// BAD
use hir::Function;
use syntax::ast::Struct;

fn frobnicate(func: Function, strukt: Struct) {}
```

**Rationale:** avoids name clashes, makes the layer clear at a glance.

When implementing traits from `std::fmt` or `std::ops`, import the module:

```rust
// GOOD
use std::fmt;

impl fmt::Display for RenameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { .. }
}

// BAD
impl std::fmt::Display for RenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { .. }
}

// BAD
use std::ops::Deref;

impl Deref for Widget {
    type Target = str;
    fn deref(&self) -> &str { .. }
}
```

**Rationale:** overall, less typing.
Makes it clear that a trait is implemented, rather than used.

Avoid local `use MyEnum::*` imports.
**Rationale:** consistency.

Prefer `use crate::foo::bar` to `use super::bar` or `use self::bar::baz`.
**Rationale:** consistency, this is the style which works in all cases.

## Order of Items

Optimize for the reader who sees the file for the first time, and wants to get a general idea about what's going on.
People read things from top to bottom, so place most important things first.

Specifically, if all items except one are private, always put the non-private item on top.

```rust
// GOOD
pub(crate) fn frobnicate() {
    Helper::act()
}

#[derive(Default)]
struct Helper { stuff: i32 }

impl Helper {
    fn act(&self) {

    }
}

// BAD
#[derive(Default)]
struct Helper { stuff: i32 }

pub(crate) fn frobnicate() {
    Helper::act()
}

impl Helper {
    fn act(&self) {

    }
}
```

If there's a mixture of private and public items, put public items first.

Put `struct`s and `enum`s first, functions and impls last. Order type declarations in top-down manner.

```rust
// GOOD
struct Parent {
    children: Vec<Child>
}

struct Child;

impl Parent {
}

impl Child {
}

// BAD
struct Child;

impl Child {
}

struct Parent {
    children: Vec<Child>
}

impl Parent {
}
```

**Rationale:** easier to get the sense of the API by visually scanning the file.
If function bodies are folded in the editor, the source code should read as documentation for the public API.

## Variable Naming

Use boring and long names for local variables ([yay code completion](https://github.com/rust-analyzer/rust-analyzer/pull/4162#discussion_r417130973)).
The default name is a lowercased name of the type: `global_state: GlobalState`.
Avoid ad-hoc acronyms and contractions, but use the ones that exist consistently (`db`, `ctx`, `acc`).
Prefer American spelling (color, behavior).

Default names:

* `res` -- "result of the function" local variable
* `it` -- I don't really care about the name
* `n_foo` -- number of foos
* `foo_idx` -- index of `foo`

Many names in rust-analyzer conflict with keywords.
We use mangled names instead of `r#ident` syntax:

```
struct -> strukt
crate  -> krate
impl   -> imp
trait  -> trait_
fn     -> func
enum   -> enum_
mod    -> module
```

**Rationale:** consistency.

## Early Returns

Do use early returns

```rust
// GOOD
fn foo() -> Option<Bar> {
    if !condition() {
        return None;
    }

    Some(...)
}

// BAD
fn foo() -> Option<Bar> {
    if condition() {
        Some(...)
    } else {
        None
    }
}
```

**Rationale:** reduce cognitive stack usage.

## Comparisons

When doing multiple comparisons use `<`/`<=`, avoid `>`/`>=`.

```rust
// GOOD
assert!(lo <= x && x <= hi);
assert!(r1 < l2 || r2 < l1);
assert!(x < y);
assert!(x > 0);

// BAD
assert!(x >= lo && x <= hi>);
assert!(r1 < l2 || l1 > r2);
assert!(y > x);
assert!(0 > x);
```

**Rationale:** Less-then comparisons are more intuitive, they correspond spatially to [real line](https://en.wikipedia.org/wiki/Real_line).

## If-let

Avoid `if let ... { } else { }` construct, use `match` instead.

```rust
// GOOD
match ctx.expected_type.as_ref() {
    Some(expected_type) => completion_ty == expected_type && !expected_type.is_unit(),
    None => false,
}

// BAD
if let Some(expected_type) = ctx.expected_type.as_ref() {
    completion_ty == expected_type && !expected_type.is_unit()
} else {
    false
}
```

**Rational:** `match` is almost always more compact.
The `else` branch can get a more precise pattern: `None` or `Err(_)` instead of `_`.

## Token names

Use `T![foo]` instead of `SyntaxKind::FOO_KW`.

```rust
// GOOD
match p.current() {
    T![true] | T![false] => true,
    _ => false,
}

// BAD

match p.current() {
    SyntaxKind::TRUE_KW | SyntaxKind::FALSE_KW => true,
    _ => false,
}
```

**Rationale:** The macro uses the familiar Rust syntax, avoiding ambiguities like "is this a brace or bracket?".

## Documentation

For `.md` and `.adoc` files, prefer a sentence-per-line format, don't wrap lines.
If the line is too long, you want to split the sentence in two :-)

**Rationale:** much easier to edit the text and read the diff.
