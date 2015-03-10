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

## Examples

### Heterogeneous Lists

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

The HList append operation is provided as an example. type macros are
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

### Additional Examples ###

#### Type-level numerics

Type-level numerics are another area where type macros can be
useful. The more common unary encodings (Peano numerals) are not
efficient enough to use in practice so we present an example
demonstrating binary natural numbers instead:

```rust
struct _0; // 0 bit
struct _1; // 1 bit

// classify valid bits
trait Bit: MarkerTrait {}
impl Bit for _0 {}
impl Bit for _1 {}

// classify positive binary naturals
trait Pos: MarkerTrait {}
impl Pos for _1 {}
impl<B: Bit, P: Pos> Pos for (P, B) {}

// classify binary naturals with 0
trait Nat: MarkerTrait {}
impl Nat for _0 {}
impl Nat for _1 {}
impl<B: Bit, P: Pos> Nat for (P, B) {}
```

These can be used to index into tuples or HLists generically, either
by specifying the path explicitly (e.g., `(a, b, c).at::<(_1, _0)>()
==> c`) or by providing a singleton term with the appropriate type
`(a, b, c).at((_1, _0)) ==> c`. Indexing is linear time in the general
case due to recursion, but can be made constant time for a fixed
number of specialized implementations.

Type-level numbers can also be used to define "sized" or "bounded"
data, such as a vector indexed by its length:

```rust
struct LengthVec<A, N: Nat>(Vec<A>);
```

Similar to the indexing example, the parameter `N` can either serve as
phantom data, or such a struct could also include a term-level
representation of N as another field.

In either case, a length-safe API could be defined for container types
like `Vec`. "Unsafe" indexing (without bounds checking) into the
underlying container would be safe in general because the length of
the container would be known statically and reflected in the type of
the length-indexed wrapper.

We could imagine an idealized API in the following fashion:

```rust
// push, adding one to the length
fn push<A, N: Nat>(xs: LengthVec<A, N>, x: A) -> LengthVec<A, N + 1>;

// pop, subtracting one from the length
fn pop<A, N: Nat>(xs: LengthVec<A, N + 1>, store: &mut A) -> LengthVec<A, N>;

// look up an element at an index
fn at<A, M: Nat, N: Nat, P: M < N>(xs: LengthVec<A, N>, index: M) -> A;

// append, adding the individual lengths
fn append<A, N: Nat, M: Nat>(xs: LengthVec<A, N>, ys: LengthVec<A, M>) -> LengthVec<A, N + M>;

// produce a length respecting iterator from an indexed vector
fn iter<A, N: Nat>(xs: LengthVec<A, N>) -> LengthIterator<A, N>;
```

We can't write code like the above directly in Rust but we could
approximate it through type-level macros:

```rust
// Expr! would expand + to Add::Output and integer constants to Nat!; see
// the HList append earlier in the RFC for a concrete example
Expr!(N + M)
    ==> <N as Add<M>>::Output

// Nat! would expand integer literals to type-level binary naturals
// and be implemented as a plugin for efficiency; see the following
// section for a concrete example
Nat!(4)
    ==> ((_1, _0), _0)

// `Expr!` and `Nat!` used for the LengthVec type:
LengthVec<A, Expr!(N + 3)>
    ==> LengthVec<A, <N as Add< Nat!(3)>>::Output>
    ==> LengthVec<A, <N as Add<(_1, _1)>>::Output>
```

##### Implementation of `Nat!` as a plugin

The following code demonstrates concretely how `Nat!` can be
implemented as a plugin. As with the `HList!` example, this code (with
some additions) compiles and is usable with the type macros prototype
in the branch referenced earlier.

For efficiency, the binary representation is first constructed as a
string via iteration rather than recursively using `quote` macros. The
string is then parsed as a type, returning an ast fragment.

```rust
// Convert a u64 to a string representation of a type-level binary natural, e.g.,
//     ast_as_str(1024)
//         ==> "(((((((((_1, _0), _0), _0), _0), _0), _0), _0), _0), _0)"
fn ast_as_str<'cx>(
        ecx: &'cx base::ExtCtxt,
    mut num: u64,
       mode: Mode,
) -> String {
    let path = "_";
    let mut res: String;
    if num < 2 {
        res = String::from_str(path);
        res.push_str(num.to_string().as_slice());
    } else {
        let mut bin = vec![];
        while num > 0 {
            bin.push(num % 2);
            num >>= 1;
        }
        res = ::std::iter::repeat('(').take(bin.len() - 1).collect();
        res.push_str(path);
        res.push_str(bin.pop().unwrap().to_string().as_slice());
        for b in bin.iter().rev() {
            res.push_str(", ");
            res.push_str(path);
            res.push_str(b.to_string().as_slice());
            res.push_str(")");
        }
    }
    res
}

// Generate a parser which uses the nat's ast-as-string as its input
fn ast_parser<'cx>(
    ecx: &'cx base::ExtCtxt,
    num: u64,
   mode: Mode,
) -> parse::parser::Parser<'cx> {
    let filemap = ecx
        .codemap()
        .new_filemap(String::from_str("<nat!>"), ast_as_str(ecx, num, mode));
    let reader  = lexer::StringReader::new(
        &ecx.parse_sess().span_diagnostic,
        filemap);
    parser::Parser::new(
        ecx.parse_sess(),
        ecx.cfg(),
        Box::new(reader))
}

// Try to parse an integer literal and return a new parser which uses
// the nat's ast-as-string as its input
pub fn lit_parser<'cx>(
     ecx: &'cx base::ExtCtxt,
    args: &[ast::TokenTree],
    mode: Mode,
) -> Option<parse::parser::Parser<'cx>> {
    let mut lit_parser = ecx.new_parser_from_tts(args);
    if let ast::Lit_::LitInt(lit, _) = lit_parser.parse_lit().node {
        Some(ast_parser(ecx, lit, mode))
    } else {
        None
    }
}

// Expand Nat!(n) to a type-level binary nat where n is an int literal, e.g.,
//     Nat!(1024)
//         ==> (((((((((_1, _0), _0), _0), _0), _0), _0), _0), _0), _0)
pub fn expand_ty<'cx>(
     ecx: &'cx mut base::ExtCtxt,
    span: codemap::Span,
    args: &[ast::TokenTree],
) -> Box<base::MacResult + 'cx> {
    {
        lit_parser(ecx, args, Mode::Ty)
    }.and_then(|mut ast_parser| {
        Some(base::MacEager::ty(ast_parser.parse_ty()))
    }).unwrap_or_else(|| {
        ecx.span_err(span, "Nat!: expected an integer literal argument");
        base::DummyResult::any(span)
    })
}

// Expand nat!(n) to a term-level binary nat where n is an int literal, e.g.,
//     nat!(1024)
//         ==> (((((((((_1, _0), _0), _0), _0), _0), _0), _0), _0), _0)
pub fn expand_tm<'cx>(
     ecx: &'cx mut base::ExtCtxt,
    span: codemap::Span,
    args: &[ast::TokenTree],
) -> Box<base::MacResult + 'cx> {
    {
        lit_parser(ecx, args, Mode::Tm)
    }.and_then(|mut ast_parser| {
        Some(base::MacEager::expr(ast_parser.parse_expr()))
    }).unwrap_or_else(|| {
        ecx.span_err(span, "nat!: expected an integer literal argument");
        base::DummyResult::any(span)
    })
}

#[test]
fn nats() {
    let _: Nat!(42) = nat!(42);
}
```

##### Optimization of `Expr`!

Defining `Expr!` as a plugin would provide an opportunity to perform
various optimizations of more complex type-level expressions during
expansion. Partial evaluation would be one way to achieve
this. Furthermore, expansion-time optimizations wouldn't be limited to
arithmetic expressions but could be used for other data like HLists.

##### Builtin alternatives: types parameterized by constant values

The example with type-level naturals serves to illustrate some of the
patterns type macros enable. This RFC is not intended to address the
lack of constant value type parameterization and type-level numerics
specifically. There is
[another RFC here](https://github.com/rust-lang/rfcs/pull/884) which
proposes extending the type system to address those issue.

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
