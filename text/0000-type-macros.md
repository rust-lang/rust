- Feature Name: Macros in type positions
- Start Date: 2015-02-16
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Allow macros in type positions

# Motivation

Macros are currently allowed in syntax fragments for expressions,
items, and patterns, but not for types. This RFC proposes to lift that
restriction.

1. This would allow macros to be used more flexibly, avoiding the
  need for more complex item-level macros or plugins in some
  cases. For example, when creating trait implementations with
  macros, it is sometimes useful to be able to define the
  associated types using a nested type macro but this is
  currently problematic.

2. Enable more programming patterns, particularly with respect to
  type level programming. Macros in type positions provide
  convenient way to express recursion and choice. It is possible
  to do the same thing purely through programming with associated
  types but the resulting code can be cumbersome to read and write.


# Detailed design

## Implementation

The proposed feature has been prototyped at
[this branch](https://github.com/freebroccolo/rust/commits/feature/type_macros). The
implementation is straightforward and the impact of the changes are
limited in scope to the macro system. Type-checking and other phases
of compilation should be unaffected.

The most significant change introduced by this feature is a
[`TyMac`](https://github.com/freebroccolo/rust/blob/f8f8dbb6d332c364ecf26b248ce5f872a7a67019/src/libsyntax/ast.rs#L1274-L1275)
case for the `Ty_` enum so that the parser can indicate a macro
invocation in a type position. In other words, `TyMac` is added to the
ast and handled analogously to `ExprMac`, `ItemMac`, and `PatMac`.

## Example: Heterogeneous Lists

Heterogeneous lists are one example where the ability to express
recursion via type macros is very useful. They can be used as an
alternative to or in combination with tuples. Their recursive
structure provide a means to abstract over arity and to manipulate
arbitrary products of types with operations like appending, taking
length, adding/removing items, computing permutations, etc.

Heterogeneous lists can be defined like so:

```rust
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Nil; // empty HList
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Cons<H, T: HList>(H, T); // cons cell of HList

// trait to classify valid HLists
trait HList: MarkerTrait {}
impl HList for Nil {}
impl<H, T: HList> HList for Cons<H, T> {}
```

However, writing HList terms in code is not very convenient:

```rust
let xs = Cons("foo", Cons(false, Cons(vec![0u64], Nil)));
```

At the term-level, this is an easy fix using macros:

```rust
// term-level macro for HLists
macro_rules! hlist {
    {} => { Nil };
    {=> $($elem:tt),+ } => { hlist_pat!($($elem),+) };
    { $head:expr, $($tail:expr),* } => { Cons($head, hlist!($($tail),*)) };
    { $head:expr } => { Cons($head, Nil) };
}

// term-level HLists in patterns
macro_rules! hlist_pat {
    {} => { Nil };
    { $head:pat, $($tail:tt),* } => { Cons($head, hlist_pat!($($tail),*)) };
    { $head:pat } => { Cons($head, Nil) };
}

let xs = hlist!["foo", false, vec![0u64]];
```

Unfortunately, this solution is incomplete because we have only made
HList terms easier to write. HList types are still inconvenient:

```rust
let xs: Cons<&str, Cons<bool, Cons<Vec<u64>, Nil>>> = hlist!["foo", false, vec![0u64]];
```

Allowing type macros as this RFC proposes would allows us to be
able to use Rust's macros to improve writing the HList type as
well. The complete example follows:

```rust
// term-level macro for HLists
macro_rules! hlist {
    {} => { Nil };
    {=> $($elem:tt),+ } => { hlist_pat!($($elem),+) };
    { $head:expr, $($tail:expr),* } => { Cons($head, hlist!($($tail),*)) };
    { $head:expr } => { Cons($head, Nil) };
}

// term-level HLists in patterns
macro_rules! hlist_pat {
    {} => { Nil };
    { $head:pat, $($tail:tt),* } => { Cons($head, hlist_pat!($($tail),*)) };
    { $head:pat } => { Cons($head, Nil) };
}

// type-level macro for HLists
macro_rules! HList {
    {} => { Nil };
    { $head:ty } => { Cons<$head, Nil> };
    { $head:ty, $($tail:ty),* } => { Cons<$head, HList!($($tail),*)> };
}

let xs: HList![&str, bool, Vec<u64>] = hlist!["foo", false, vec![0u64]];
```

Operations on HLists can be defined by recursion, using traits with
associated type outputs at the type-level and implementation methods
at the term-level.

The HList append operation is provided as an example. Type macros are
used to make writing append at the type level (see `Expr!`) more
convenient than specifying the associated type projection manually:

```rust
use std::ops;

// nil case for HList append
impl<Ys: HList> ops::Add<Ys> for Nil {
    type Output = Ys;

    fn add(self, rhs: Ys) -> Ys {
        rhs
    }
}

// cons case for HList append
impl<Rec: HList + Sized, X, Xs: HList, Ys: HList> ops::Add<Ys> for Cons<X, Xs> where
    Xs: ops::Add<Ys, Output = Rec>,
{
    type Output = Cons<X, Rec>;

    fn add(self, rhs: Ys) -> Cons<X, Rec> {
        Cons(self.0, self.1 + rhs)
    }
}

// type macro Expr allows us to expand the + operator appropriately
macro_rules! Expr {
    { ( $($LHS:tt)+ ) } => { Expr!($($LHS)+) };
    { HList ! [ $($LHS:tt)* ] + $($RHS:tt)+ } => { <Expr!(HList![$($LHS)*]) as std::ops::Add<Expr!($($RHS)+)>>::Output };
    { $LHS:tt + $($RHS:tt)+ } => { <Expr!($LHS) as std::ops::Add<Expr!($($RHS)+)>>::Output };
    { $LHS:ty } => { $LHS };
}

// test demonstrating term level `xs + ys` and type level `Expr!(Xs + Ys)`
#[test]
fn test_append() {
    fn aux<Xs: HList, Ys: HList>(xs: Xs, ys: Ys) -> Expr!(Xs + Ys) where
        Xs: ops::Add<Ys>
    {
        xs + ys
    }
    let xs: HList![&str, bool, Vec<u64>] = hlist!["foo", false, vec![]];
    let ys: HList![u64, [u8; 3], ()] = hlist![0, [0, 1, 2], ()];

    // demonstrate recursive expansion of Expr!
    let zs: Expr!((HList![&str] + HList![bool] + HList![Vec<u64>]) +
                  (HList![u64] + HList![[u8; 3], ()]) +
                  HList![])
        = aux(xs, ys);
    assert_eq!(zs, hlist!["foo", false, vec![], 0, [0, 1, 2], ()])
}
```

# Drawbacks

There seem to be few drawbacks to implementing this feature as an
extension of the existing macro machinery. The change adds a small
amount of additional complexity to the
[parser](https://github.com/freebroccolo/rust/commit/a224739e92a3aa1febb67d6371988622bd141361)
and
[conversion](https://github.com/freebroccolo/rust/commit/9341232087991dee73713dc4521acdce11a799a2)
but the changes are minimal.

As with all feature proposals, it is possible that designs for future
extensions to the macro system or type system might interfere with
this functionality but it seems unlikely unless they are significant,
breaking changes.

# Alternatives

There are no _direct_ alternatives. Extensions to the type system like
data kinds, singletons, and other forms of staged programming
(so-called CTFE) might alleviate the need for type macros in some
cases, however it is unlikely that they would provide a comprehensive
replacement, particularly where plugins are concerned.

Not implementing this feature would mean not taking some reasonably
low-effort steps toward making certain programming patterns
easier. One potential consequence of this might be more pressure to
significantly extend the type system and other aspects of the language
to compensate.

# Unresolved questions

## Alternative syntax for macro invocations in types

There is a question as to whether type macros should allow `<` and `>`
as delimiters for invocations, e.g. `Foo!<A>`. This would raise a
number of additional complications and is probably not necessary to
consider for this RFC. If deemed desirable by the community, this
functionality should be proposed separately.

## Hygiene and type macros

This RFC also does not address the topic of hygiene regarding macros
in types. It is not clear whether there are issues here or not but it
may be worth considering in further detail.
