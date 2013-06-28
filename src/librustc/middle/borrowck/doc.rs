// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# The Borrow Checker

This pass has the job of enforcing memory safety. This is a subtle
topic. This docs aim to explain both the practice and the theory
behind the borrow checker. They start with a high-level overview of
how it works, and then proceed to dive into the theoretical
background. Finally, they go into detail on some of the more subtle
aspects.

# Table of contents

These docs are long. Search for the section you are interested in.

- Overview
- Formal model
- Borrowing and loans
- Moves and initialization
- Future work

# Overview

The borrow checker checks one function at a time. It operates in two
passes. The first pass, called `gather_loans`, walks over the function
and identifies all of the places where borrows (e.g., `&` expressions
and `ref` bindings) and moves (copies or captures of a linear value)
occur. It also tracks initialization sites. For each borrow and move,
it checks various basic safety conditions at this time (for example,
that the lifetime of the borrow doesn't exceed the lifetime of the
value being borrowed, or that there is no move out of an `&T`
pointee).

It then uses the dataflow module to propagate which of those borrows
may be in scope at each point in the procedure. A loan is considered
to come into scope at the expression that caused it and to go out of
scope when the lifetime of the resulting borrowed pointer expires.

Once the in-scope loans are known for each point in the program, the
borrow checker walks the IR again in a second pass called
`check_loans`. This pass examines each statement and makes sure that
it is safe with respect to the in-scope loans.

# Formal model

Throughout the docs we'll consider a simple subset of Rust in which
you can only borrow from lvalues, defined like so:

    LV = x | LV.f | *LV

Here `x` represents some variable, `LV.f` is a field reference,
and `*LV` is a pointer dereference. There is no auto-deref or other
niceties. This means that if you have a type like:

    struct S { f: uint }

and a variable `a: ~S`, then the rust expression `a.f` would correspond
to an `LV` of `(*a).f`.

Here is the formal grammar for the types we'll consider:

    TY = () | S<'LT...> | ~TY | & 'LT MQ TY | @ MQ TY
    MQ = mut | imm | const

Most of these types should be pretty self explanatory. Here `S` is a
struct name and we assume structs are declared like so:

    SD = struct S<'LT...> { (f: TY)... }

# Borrowing and loans

## An intuitive explanation

### Issuing loans

Now, imagine we had a program like this:

    struct Foo { f: uint, g: uint }
    ...
    'a: {
      let mut x: ~Foo = ...;
      let y = &mut (*x).f;
      x = ...;
    }

This is of course dangerous because mutating `x` will free the old
value and hence invalidate `y`. The borrow checker aims to prevent
this sort of thing.

#### Loans and restrictions

The way the borrow checker works is that it analyzes each borrow
expression (in our simple model, that's stuff like `&LV`, though in
real life there are a few other cases to consider). For each borrow
expression, it computes a `Loan`, which is a data structure that
records (1) the value being borrowed, (2) the mutability and scope of
the borrow, and (3) a set of restrictions. In the code, `Loan` is a
struct defined in `middle::borrowck`. Formally, we define `LOAN` as
follows:

    LOAN = (LV, LT, MQ, RESTRICTION*)
    RESTRICTION = (LV, ACTION*)
    ACTION = MUTATE | CLAIM | FREEZE | ALIAS

Here the `LOAN` tuple defines the lvalue `LV` being borrowed; the
lifetime `LT` of that borrow; the mutability `MQ` of the borrow; and a
list of restrictions. The restrictions indicate actions which, if
taken, could invalidate the loan and lead to type safety violations.

Each `RESTRICTION` is a pair of a restrictive lvalue `LV` (which will
either be the path that was borrowed or some prefix of the path that
was borrowed) and a set of restricted actions.  There are three kinds
of actions that may be restricted for the path `LV`:

- `MUTATE` means that `LV` cannot be assigned to;
- `CLAIM` means that the `LV` cannot be borrowed mutably;
- `FREEZE` means that the `LV` cannot be borrowed immutably;
- `ALIAS` means that `LV` cannot be aliased in any way (not even `&const`).

Finally, it is never possible to move from an lvalue that appears in a
restriction. This implies that the "empty restriction" `(LV, [])`,
which contains an empty set of actions, still has a purpose---it
prevents moves from `LV`. I chose not to make `MOVE` a fourth kind of
action because that would imply that sometimes moves are permitted
from restrictived values, which is not the case.

#### Example

To give you a better feeling for what kind of restrictions derived
from a loan, let's look at the loan `L` that would be issued as a
result of the borrow `&mut (*x).f` in the example above:

    L = ((*x).f, 'a, mut, RS) where
        RS = [((*x).f, [MUTATE, CLAIM, FREEZE]),
              (*x, [MUTATE, CLAIM, FREEZE]),
              (x, [MUTATE, CLAIM, FREEZE])]

The loan states that the expression `(*x).f` has been loaned as
mutable for the lifetime `'a`. Because the loan is mutable, that means
that the value `(*x).f` may be mutated via the newly created borrowed
pointer (and *only* via that pointer). This is reflected in the
restrictions `RS` that accompany the loan.

The first restriction `((*x).f, [MUTATE, CLAIM, FREEZE])` states that
the lender may not mutate nor freeze `(*x).f`. Mutation is illegal
because `(*x).f` is only supposed to be mutated via the new borrowed
pointer, not by mutating the original path `(*x).f`. Freezing is
illegal because the path now has an `&mut` alias; so even if we the
lender were to consider `(*x).f` to be immutable, it might be mutated
via this alias. Both of these restrictions are temporary. They will be
enforced for the lifetime `'a` of the loan. After the loan expires,
the restrictions no longer apply.

The second restriction on `*x` is interesting because it does not
apply to the path that was lent (`(*x).f`) but rather to a prefix of
the borrowed path. This is due to the rules of inherited mutability:
if the user were to assign to (or freeze) `*x`, they would indirectly
overwrite (or freeze) `(*x).f`, and thus invalidate the borrowed
pointer that was created. In general it holds that when a path is
lent, restrictions are issued for all the owning prefixes of that
path. In this case, the path `*x` owns the path `(*x).f` and,
because `x` is an owned pointer, the path `x` owns the path `*x`.
Therefore, borrowing `(*x).f` yields restrictions on both
`*x` and `x`.

### Checking for illegal assignments, moves, and reborrows

Once we have computed the loans introduced by each borrow, the borrow
checker uses a data flow propagation to compute the full set of loans
in scope at each expression and then uses that set to decide whether
that expression is legal.  Remember that the scope of loan is defined
by its lifetime LT.  We sometimes say that a loan which is in-scope at
a particular point is an "outstanding loan", aand the set of
restrictions included in those loans as the "outstanding
restrictions".

The kinds of expressions which in-scope loans can render illegal are:
- *assignments* (`lv = v`): illegal if there is an in-scope restriction
  against mutating `lv`;
- *moves*: illegal if there is any in-scope restriction on `lv` at all;
- *mutable borrows* (`&mut lv`): illegal there is an in-scope restriction
  against mutating `lv` or aliasing `lv`;
- *immutable borrows* (`&lv`): illegal there is an in-scope restriction
  against freezing `lv` or aliasing `lv`;
- *read-only borrows* (`&const lv`): illegal there is an in-scope restriction
  against aliasing `lv`.

## Formal rules

Now that we hopefully have some kind of intuitive feeling for how the
borrow checker works, let's look a bit more closely now at the precise
conditions that it uses. For simplicity I will ignore const loans.

I will present the rules in a modified form of standard inference
rules, which looks as as follows:

    PREDICATE(X, Y, Z)                  // Rule-Name
      Condition 1
      Condition 2
      Condition 3

The initial line states the predicate that is to be satisfied.  The
indented lines indicate the conditions that must be met for the
predicate to be satisfied. The right-justified comment states the name
of this rule: there are comments in the borrowck source referencing
these names, so that you can cross reference to find the actual code
that corresponds to the formal rule.

### The `gather_loans` pass

We start with the `gather_loans` pass, which walks the AST looking for
borrows.  For each borrow, there are three bits of information: the
lvalue `LV` being borrowed and the mutability `MQ` and lifetime `LT`
of the resulting pointer. Given those, `gather_loans` applies three
validity tests:

1. `MUTABILITY(LV, MQ)`: The mutability of the borrowed pointer is
compatible with the mutability of `LV` (i.e., not borrowing immutable
data as mutable).

2. `LIFETIME(LV, LT, MQ)`: The lifetime of the borrow does not exceed
the lifetime of the value being borrowed. This pass is also
responsible for inserting root annotations to keep managed values
alive and for dynamically freezing `@mut` boxes.

3. `RESTRICTIONS(LV, ACTIONS) = RS`: This pass checks and computes the
restrictions to maintain memory safety. These are the restrictions
that will go into the final loan. We'll discuss in more detail below.

## Checking mutability

Checking mutability is fairly straightforward. We just want to prevent
immutable data from being borrowed as mutable. Note that it is ok to
borrow mutable data as immutable, since that is simply a
freeze. Formally we define a predicate `MUTABLE(LV, MQ)` which, if
defined, means that "borrowing `LV` with mutability `MQ` is ok. The
Rust code corresponding to this predicate is the function
`check_mutability` in `middle::borrowck::gather_loans`.

### Checking mutability of variables

*Code pointer:* Function `check_mutability()` in `gather_loans/mod.rs`,
but also the code in `mem_categorization`.

Let's begin with the rules for variables, which state that if a
variable is declared as mutable, it may be borrowed any which way, but
otherwise the variable must be borrowed as immutable or const:

    MUTABILITY(X, MQ)                   // M-Var-Mut
      DECL(X) = mut

    MUTABILITY(X, MQ)                   // M-Var-Imm
      DECL(X) = imm
      MQ = imm | const

### Checking mutability of owned content

Fields and owned pointers inherit their mutability from
their base expressions, so both of their rules basically
delegate the check to the base expression `LV`:

    MUTABILITY(LV.f, MQ)                // M-Field
      MUTABILITY(LV, MQ)

    MUTABILITY(*LV, MQ)                 // M-Deref-Unique
      TYPE(LV) = ~Ty
      MUTABILITY(LV, MQ)

### Checking mutability of immutable pointer types

Immutable pointer types like `&T` and `@T` can only
be borrowed if MQ is immutable or const:

    MUTABILITY(*LV, MQ)                // M-Deref-Borrowed-Imm
      TYPE(LV) = &Ty
      MQ == imm | const

    MUTABILITY(*LV, MQ)                // M-Deref-Managed-Imm
      TYPE(LV) = @Ty
      MQ == imm | const

### Checking mutability of mutable pointer types

`&mut T` and `@mut T` can be frozen, so it is acceptable to borrow
them as either imm or mut:

    MUTABILITY(*LV, MQ)                 // M-Deref-Borrowed-Mut
      TYPE(LV) = &mut Ty

    MUTABILITY(*LV, MQ)                 // M-Deref-Managed-Mut
      TYPE(LV) = @mut Ty

## Checking lifetime

These rules aim to ensure that no data is borrowed for a scope that
exceeds its lifetime. In addition, these rules manage the rooting and
dynamic freezing of `@` and `@mut` values. These two computations wind
up being intimately related. Formally, we define a predicate
`LIFETIME(LV, LT, MQ)`, which states that "the lvalue `LV` can be
safely borrowed for the lifetime `LT` with mutability `MQ`". The Rust
code corresponding to this predicate is the module
`middle::borrowck::gather_loans::lifetime`.

### The Scope function

Several of the rules refer to a helper function `SCOPE(LV)=LT`.  The
`SCOPE(LV)` yields the lifetime `LT` for which the lvalue `LV` is
guaranteed to exist, presuming that no mutations occur.

The scope of a local variable is the block where it is declared:

      SCOPE(X) = block where X is declared

The scope of a field is the scope of the struct:

      SCOPE(LV.f) = SCOPE(LV)

The scope of a unique pointee is the scope of the pointer, since
(barring mutation or moves) the pointer will not be freed until
the pointer itself `LV` goes out of scope:

      SCOPE(*LV) = SCOPE(LV) if LV has type ~T

The scope of a managed pointee is also the scope of the pointer.  This
is a conservative approximation, since there may be other aliases fo
that same managed box that would cause it to live longer:

      SCOPE(*LV) = SCOPE(LV) if LV has type @T or @mut T

The scope of a borrowed pointee is the scope associated with the
pointer.  This is a conservative approximation, since the data that
the pointer points at may actually live longer:

      SCOPE(*LV) = LT if LV has type &'LT T or &'LT mut T

### Checking lifetime of variables

The rule for variables states that a variable can only be borrowed a
lifetime `LT` that is a subregion of the variable's scope:

    LIFETIME(X, LT, MQ)                 // L-Local
      LT <= SCOPE(X)

### Checking lifetime for owned content

The lifetime of a field or owned pointer is the same as the lifetime
of its owner:

    LIFETIME(LV.f, LT, MQ)              // L-Field
      LIFETIME(LV, LT, MQ)

    LIFETIME(*LV, LT, MQ)               // L-Deref-Send
      TYPE(LV) = ~Ty
      LIFETIME(LV, LT, MQ)

### Checking lifetime for derefs of borrowed pointers

Borrowed pointers have a lifetime `LT'` associated with them.  The
data they point at has been guaranteed to be valid for at least this
lifetime. Therefore, the borrow is valid so long as the lifetime `LT`
of the borrow is shorter than the lifetime `LT'` of the pointer
itself:

    LIFETIME(*LV, LT, MQ)               // L-Deref-Borrowed
      TYPE(LV) = &LT' Ty OR &LT' mut Ty
      LT <= LT'

### Checking lifetime for derefs of managed, immutable pointers

Managed pointers are valid so long as the data within them is
*rooted*. There are two ways that this can be achieved. The first is
when the user guarantees such a root will exist. For this to be true,
three conditions must be met:

    LIFETIME(*LV, LT, MQ)               // L-Deref-Managed-Imm-User-Root
      TYPE(LV) = @Ty
      LT <= SCOPE(LV)                   // (1)
      LV is immutable                   // (2)
      LV is not moved or not movable    // (3)

Condition (1) guarantees that the managed box will be rooted for at
least the lifetime `LT` of the borrow, presuming that no mutation or
moves occur. Conditions (2) and (3) then serve to guarantee that the
value is not mutated or moved. Note that lvalues are either
(ultimately) owned by a local variable, in which case we can check
whether that local variable is ever moved in its scope, or they are
owned by the pointee of an (immutable, due to condition 2) managed or
borrowed pointer, in which case moves are not permitted because the
location is aliasable.

If the conditions of `L-Deref-Managed-Imm-User-Root` are not met, then
there is a second alternative. The compiler can attempt to root the
managed pointer itself. This permits great flexibility, because the
location `LV` where the managed pointer is found does not matter, but
there are some limitations. The lifetime of the borrow can only extend
to the innermost enclosing loop or function body. This guarantees that
the compiler never requires an unbounded amount of stack space to
perform the rooting; if this condition were violated, the compiler
might have to accumulate a list of rooted objects, for example if the
borrow occurred inside the body of a loop but the scope of the borrow
extended outside the loop. More formally, the requirement is that
there is no path starting from the borrow that leads back to the
borrow without crossing the exit from the scope `LT`.

The rule for compiler rooting is as follows:

    LIFETIME(*LV, LT, MQ)               // L-Deref-Managed-Imm-Compiler-Root
      TYPE(LV) = @Ty
      LT <= innermost enclosing loop/func
      ROOT LV at *LV for LT

Here I have written `ROOT LV at *LV FOR LT` to indicate that the code
makes a note in a side-table that the box `LV` must be rooted into the
stack when `*LV` is evaluated, and that this root can be released when
the scope `LT` exits.

### Checking lifetime for derefs of managed, mutable pointers

Loans of the contents of mutable managed pointers are simpler in some
ways that loans of immutable managed pointers, because we can never
rely on the user to root them (since the contents are, after all,
mutable). This means that the burden always falls to the compiler, so
there is only one rule:

    LIFETIME(*LV, LT, MQ)              // L-Deref-Managed-Mut-Compiler-Root
      TYPE(LV) = @mut Ty
      LT <= innermost enclosing loop/func
      ROOT LV at *LV for LT
      LOCK LV at *LV as MQ for LT

Note that there is an additional clause this time `LOCK LV at *LV as
MQ for LT`.  This clause states that in addition to rooting `LV`, the
compiler should also "lock" the box dynamically, meaning that we
register that the box has been borrowed as mutable or immutable,
depending on `MQ`. This lock will fail if the box has already been
borrowed and either the old loan or the new loan is a mutable loan
(multiple immutable loans are okay). The lock is released as we exit
the scope `LT`.

## Computing the restrictions

The final rules govern the computation of *restrictions*, meaning that
we compute the set of actions that will be illegal for the life of the
loan. The predicate is written `RESTRICTIONS(LV, ACTIONS) =
RESTRICTION*`, which can be read "in order to prevent `ACTIONS` from
occuring on `LV`, the restrictions `RESTRICTION*` must be respected
for the lifetime of the loan".

Note that there is an initial set of restrictions: these restrictions
are computed based on the kind of borrow:

    &mut LV =>   RESTRICTIONS(LV, MUTATE|CLAIM|FREEZE)
    &LV =>       RESTRICTIONS(LV, MUTATE|CLAIM)
    &const LV => RESTRICTIONS(LV, [])

The reasoning here is that a mutable borrow must be the only writer,
therefore it prevents other writes (`MUTATE`), mutable borrows
(`CLAIM`), and immutable borrows (`FREEZE`). An immutable borrow
permits other immutable borows but forbids writes and mutable borows.
Finally, a const borrow just wants to be sure that the value is not
moved out from under it, so no actions are forbidden.

### Restrictions for loans of a local variable

The simplest case is a borrow of a local variable `X`:

    RESTRICTIONS(X, ACTIONS) = (X, ACTIONS)            // R-Variable

In such cases we just record the actions that are not permitted.

### Restrictions for loans of fields

Restricting a field is the same as restricting the owner of that
field:

    RESTRICTIONS(LV.f, ACTIONS) = RS, (LV.f, ACTIONS)  // R-Field
      RESTRICTIONS(LV, ACTIONS) = RS

The reasoning here is as follows. If the field must not be mutated,
then you must not mutate the owner of the field either, since that
would indirectly modify the field. Similarly, if the field cannot be
frozen or aliased, we cannot allow the owner to be frozen or aliased,
since doing so indirectly freezes/aliases the field. This is the
origin of inherited mutability.

### Restrictions for loans of owned pointees

Because the mutability of owned pointees is inherited, restricting an
owned pointee is similar to restricting a field, in that it implies
restrictions on the pointer. However, owned pointers have an important
twist: if the owner `LV` is mutated, that causes the owned pointee
`*LV` to be freed! So whenever an owned pointee `*LV` is borrowed, we
must prevent the owned pointer `LV` from being mutated, which means
that we always add `MUTATE` and `CLAIM` to the restriction set imposed
on `LV`:

    RESTRICTIONS(*LV, ACTIONS) = RS, (*LV, ACTIONS)    // R-Deref-Send-Pointer
      TYPE(LV) = ~Ty
      RESTRICTIONS(LV, ACTIONS|MUTATE|CLAIM) = RS

### Restrictions for loans of immutable managed/borrowed pointees

Immutable managed/borrowed pointees are freely aliasable, meaning that
the compiler does not prevent you from copying the pointer.  This
implies that issuing restrictions is useless. We might prevent the
user from acting on `*LV` itself, but there could be another path
`*LV1` that refers to the exact same memory, and we would not be
restricting that path. Therefore, the rule for `&Ty` and `@Ty`
pointers always returns an empty set of restrictions, and it only
permits restricting `MUTATE` and `CLAIM` actions:

    RESTRICTIONS(*LV, ACTIONS) = []                    // R-Deref-Imm-Borrowed
      TYPE(LV) = &Ty or @Ty
      ACTIONS subset of [MUTATE, CLAIM]

The reason that we can restrict `MUTATE` and `CLAIM` actions even
without a restrictions list is that it is never legal to mutate nor to
borrow mutably the contents of a `&Ty` or `@Ty` pointer. In other
words, those restrictions are already inherent in the type.

Typically, this limitation is not an issue, because restrictions other
than `MUTATE` or `CLAIM` typically arise due to `&mut` borrow, and as
we said, that is already illegal for `*LV`. However, there is one case
where we can be asked to enforce an `ALIAS` restriction on `*LV`,
which is when you have a type like `&&mut T`. In such cases we will
report an error because we cannot enforce a lack of aliases on a `&Ty`
or `@Ty` type. That case is described in more detail in the section on
mutable borrowed pointers.

### Restrictions for loans of const aliasable pointees

Freeze pointers are read-only. There may be `&mut` or `&` aliases, and
we can not prevent *anything* but moves in that case. So the
`RESTRICTIONS` function is only defined if `ACTIONS` is the empty set.
Because moves from a `&const` or `@const` lvalue are never legal, it
is not necessary to add any restrictions at all to the final
result.

    RESTRICTIONS(*LV, []) = []                         // R-Deref-Freeze-Borrowed
      TYPE(LV) = &const Ty or @const Ty

### Restrictions for loans of mutable borrowed pointees

Borrowing mutable borrowed pointees is a bit subtle because we permit
users to freeze or claim `&mut` pointees. To see what I mean, consider this
(perfectly safe) code example:

    fn foo(t0: &mut T, op: fn(&T)) {
        let t1: &T = &*t0; // (1)
        op(t1);
    }

In the borrow marked `(1)`, the data at `*t0` is *frozen* as part of a
re-borrow. Therefore, for the lifetime of `t1`, `*t0` must not be
mutated. This is the same basic idea as when we freeze a mutable local
variable, but unlike in that case `t0` is a *pointer* to the data, and
thus we must enforce some subtle restrictions in order to guarantee
soundness.

Intuitively, we must ensure that `*t0` is the only *mutable* path to
reach the memory that was frozen. The reason that we are so concerned
with *mutable* paths is that those are the paths through which the
user could mutate the data that was frozen and hence invalidate the
`t1` pointer. Note that const aliases to `*t0` are acceptable (and in
fact we can't prevent them without unacceptable performance cost, more
on that later) because

There are two rules governing `&mut` pointers, but we'll begin with
the first. This rule governs cases where we are attempting to prevent
an `&mut` pointee from being mutated, claimed, or frozen, as occurs
whenever the `&mut` pointee `*LV` is reborrowed as mutable or
immutable:

    RESTRICTIONS(*LV, ACTIONS) = RS, (*LV, ACTIONS)    // R-Deref-Mut-Borrowed-1
      TYPE(LV) = &mut Ty
      RESTRICTIONS(LV, MUTATE|CLAIM|ALIAS) = RS

The main interesting part of the rule is the final line, which
requires that the `&mut` *pointer* `LV` be restricted from being
mutated, claimed, or aliased. The goal of these restrictions is to
ensure that, not considering the pointer that will result from this
borrow, `LV` remains the *sole pointer with mutable access* to `*LV`.

Restrictions against mutations and claims are necessary because if the
pointer in `LV` were to be somehow copied or moved to a different
location, then the restriction issued for `*LV` would not apply to the
new location. Note that because `&mut` values are non-copyable, a
simple attempt to move the base pointer will fail due to the
(implicit) restriction against moves:

    // src/test/compile-fail/borrowck-move-mut-base-ptr.rs
    fn foo(t0: &mut int) {
        let p: &int = &*t0; // Freezes `*t0`
        let t1 = t0;        //~ ERROR cannot move out of `t0`
        *t1 = 22;
    }

However, the additional restrictions against mutation mean that even a
clever attempt to use a swap to circumvent the type system will
encounter an error:

    // src/test/compile-fail/borrowck-swap-mut-base-ptr.rs
    fn foo<'a>(mut t0: &'a mut int,
               mut t1: &'a mut int) {
        let p: &int = &*t0;     // Freezes `*t0`
        swap(&mut t0, &mut t1); //~ ERROR cannot borrow `t0`
        *t1 = 22;
    }

The restriction against *aliasing* (and, in turn, freezing) is
necessary because, if an alias were of `LV` were to be produced, then
`LV` would no longer be the sole path to access the `&mut`
pointee. Since we are only issuing restrictions against `*LV`, these
other aliases would be unrestricted, and the result would be
unsound. For example:

    // src/test/compile-fail/borrowck-alias-mut-base-ptr.rs
    fn foo(t0: &mut int) {
        let p: &int = &*t0; // Freezes `*t0`
        let q: &const &mut int = &const t0; //~ ERROR cannot borrow `t0`
        **q = 22; // (*)
    }

Note that the current rules also report an error at the assignment in
`(*)`, because we only permit `&mut` poiners to be assigned if they
are located in a non-aliasable location. However, I do not believe
this restriction is strictly necessary. It was added, I believe, to
discourage `&mut` from being placed in aliasable locations in the
first place. One (desirable) side-effect of restricting aliasing on
`LV` is that borrowing an `&mut` pointee found inside an aliasable
pointee yields an error:

    // src/test/compile-fail/borrowck-borrow-mut-base-ptr-in-aliasable-loc:
    fn foo(t0: & &mut int) {
        let t1 = t0;
        let p: &int = &**t0; //~ ERROR cannot borrow an `&mut` in a `&` pointer
        **t1 = 22; // (*)
    }

Here at the line `(*)` you will also see the error I referred to
above, which I do not believe is strictly necessary.

The second rule for `&mut` handles the case where we are not adding
any restrictions (beyond the default of "no move"):

    RESTRICTIONS(*LV, []) = []                    // R-Deref-Mut-Borrowed-2
      TYPE(LV) = &mut Ty

Moving from an `&mut` pointee is never legal, so no special
restrictions are needed.

### Restrictions for loans of mutable managed pointees

With `@mut` pointees, we don't make any static guarantees.  But as a
convenience, we still register a restriction against `*LV`, because
that way if we *can* find a simple static error, we will:

    RESTRICTIONS(*LV, ACTIONS) = [*LV, ACTIONS]   // R-Deref-Managed-Borrowed
      TYPE(LV) = @mut Ty

# Moves and initialization

The borrow checker is also in charge of ensuring that:

- all memory which is accessed is initialized
- immutable local variables are assigned at most once.

These are two separate dataflow analyses built on the same
framework. Let's look at checking that memory is initialized first;
the checking of immutable local variabe assignments works in a very
similar way.

To track the initialization of memory, we actually track all the
points in the program that *create uninitialized memory*, meaning
moves and the declaration of uninitialized variables. For each of
these points, we create a bit in the dataflow set. Assignments to a
variable `x` or path `a.b.c` kill the move/uninitialization bits for
those paths and any subpaths (e.g., `x`, `x.y`, `a.b.c`, `*a.b.c`).
The bits are also killed when the root variables (`x`, `a`) go out of
scope. Bits are unioned when two control-flow paths join. Thus, the
presence of a bit indicates that the move may have occurred without an
intervening assignment to the same memory. At each use of a variable,
we examine the bits in scope, and check that none of them are
moves/uninitializations of the variable that is being used.

Let's look at a simple example:

    fn foo(a: ~int) {
        let b: ~int;       // Gen bit 0.

        if cond {          // Bits: 0
            use(&*a);
            b = a;         // Gen bit 1, kill bit 0.
            use(&*b);
        } else {
                           // Bits: 0
        }
                           // Bits: 0,1
        use(&*a);          // Error.
        use(&*b);          // Error.
    }

    fn use(a: &int) { }

In this example, the variable `b` is created uninitialized. In one
branch of an `if`, we then move the variable `a` into `b`. Once we
exit the `if`, therefore, it is an error to use `a` or `b` since both
are only conditionally initialized. I have annotated the dataflow
state using comments. There are two dataflow bits, with bit 0
corresponding to the creation of `b` without an initializer, and bit 1
corresponding to the move of `a`. The assignment `b = a` both
generates bit 1, because it is a move of `a`, and kills bit 0, because
`b` is now initialized. On the else branch, though, `b` is never
initialized, and so bit 0 remains untouched. When the two flows of
control join, we union the bits from both sides, resulting in both
bits 0 and 1 being set. Thus any attempt to use `a` uncovers the bit 1
from the "then" branch, showing that `a` may be moved, and any attempt
to use `b` uncovers bit 0, from the "else" branch, showing that `b`
may not be initialized.

## Initialization of immutable variables

Initialization of immutable variables works in a very similar way,
except that:

1. we generate bits for each assignment to a variable;
2. the bits are never killed except when the variable goes out of scope.

Thus the presence of an assignment bit indicates that the assignment
may have occurred. Note that assignments are only killed when the
variable goes out of scope, as it is not relevant whether or not there
has been a move in the meantime. Using these bits, we can declare that
an assignment to an immutable variable is legal iff there is no other
assignment bit to that same variable in scope.

## Why is the design made this way?

It may seem surprising that we assign dataflow bits to *each move*
rather than *each path being moved*. This is somewhat less efficient,
since on each use, we must iterate through all moves and check whether
any of them correspond to the path in question. Similar concerns apply
to the analysis for double assignments to immutable variables. The
main reason to do it this way is that it allows us to print better
error messages, because when a use occurs, we can print out the
precise move that may be in scope, rather than simply having to say
"the variable may not be initialized".

## Data structures used in the move analysis

The move analysis maintains several data structures that enable it to
cross-reference moves and assignments to determine when they may be
moving/assigning the same memory. These are all collected into the
`MoveData` and `FlowedMoveData` structs. The former represents the set
of move paths, moves, and assignments, and the latter adds in the
results of a dataflow computation.

### Move paths

The `MovePath` tree tracks every path that is moved or assigned to.
These paths have the same form as the `LoanPath` data structure, which
in turn is the "real world version of the lvalues `LV` that we
introduced earlier. The difference between a `MovePath` and a `LoanPath`
is that move paths are:

1. Canonicalized, so that we have exactly one copy of each, and
   we can refer to move paths by index;
2. Cross-referenced with other paths into a tree, so that given a move
   path we can efficiently find all parent move paths and all
   extensions (e.g., given the `a.b` move path, we can easily find the
   move path `a` and also the move paths `a.b.c`)
3. Cross-referenced with moves and assignments, so that we can
   easily find all moves and assignments to a given path.

The mechanism that we use is to create a `MovePath` record for each
move path. These are arranged in an array and are referenced using
`MovePathIndex` values, which are newtype'd indices. The `MovePath`
structs are arranged into a tree, representing using the standard
Knuth representation where each node has a child 'pointer' and a "next
sibling" 'pointer'. In addition, each `MovePath` has a parent
'pointer'.  In this case, the 'pointers' are just `MovePathIndex`
values.

In this way, if we want to find all base paths of a given move path,
we can just iterate up the parent pointers (see `each_base_path()` in
the `move_data` module). If we want to find all extensions, we can
iterate through the subtree (see `each_extending_path()`).

### Moves and assignments

There are structs to represent moves (`Move`) and assignments
(`Assignment`), and these are also placed into arrays and referenced
by index. All moves of a particular path are arranged into a linked
lists, beginning with `MovePath.first_move` and continuing through
`Move.next_move`.

We distinguish between "var" assignments, which are assignments to a
variable like `x = foo`, and "path" assignments (`x.f = foo`).  This
is because we need to assign dataflows to the former, but not the
latter, so as to check for double initialization of immutable
variables.

### Gathering and checking moves

Like loans, we distinguish two phases. The first, gathering, is where
we uncover all the moves and assignments. As with loans, we do some
basic sanity checking in this phase, so we'll report errors if you
attempt to move out of a borrowed pointer etc. Then we do the dataflow
(see `FlowedMoveData::new`). Finally, in the `check_loans.rs` code, we
walk back over, identify all uses, assignments, and captures, and
check that they are legal given the set of dataflow bits we have
computed for that program point.

# Future work

While writing up these docs, I encountered some rules I believe to be
stricter than necessary:

- I think the restriction against mutating `&mut` pointers found in an
  aliasable location is unnecessary. They cannot be reborrowed, to be sure,
  so it should be safe to mutate them. Lifting this might cause some common
  cases (`&mut int`) to work just fine, but might lead to further confusion
  in other cases, so maybe it's best to leave it as is.
- I think restricting the `&mut` LV against moves and `ALIAS` is sufficient,
  `MUTATE` and `CLAIM` are overkill. `MUTATE` was necessary when swap was
  a built-in operator, but as it is not, it is implied by `CLAIM`,
  and `CLAIM` is implied by `ALIAS`. The only net effect of this is an
  extra error message in some cases, though.
- I have not described how closures interact. Current code is unsound.
  I am working on describing and implementing the fix.
- If we wish, we can easily extend the move checking to allow finer-grained
  tracking of what is initialized and what is not, enabling code like
  this:

      a = x.f.g; // x.f.g is now uninitialized
      // here, x and x.f are not usable, but x.f.h *is*
      x.f.g = b; // x.f.g is not initialized
      // now x, x.f, x.f.g, x.f.h are all usable

  What needs to change here, most likely, is that the `moves` module
  should record not only what paths are moved, but what expressions
  are actual *uses*. For example, the reference to `x` in `x.f.g = b`
  is not a true *use* in the sense that it requires `x` to be fully
  initialized. This is in fact why the above code produces an error
  today: the reference to `x` in `x.f.g = b` is considered illegal
  because `x` is not fully initialized.

There are also some possible refactorings:

- It might be nice to replace all loan paths with the MovePath mechanism,
  since they allow lightweight comparison using an integer.

*/
