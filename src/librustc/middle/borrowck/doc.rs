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
topic. The only way I know how to explain it is terms of a formal
model, so that's what I'll do.

# Formal model

Let's consider a simple subset of Rust in which you can only borrow
from lvalues like so:

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

# An intuitive explanation

## Issuing loans

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

### Loans

The way the borrow checker works is that it analyzes each borrow
expression (in our simple model, that's stuff like `&LV`, though in
real life there are a few other cases to consider). For each borrow
expression, it computes a vector of loans:

    LOAN = (LV, LT, PT, LK)
    PT = Partial | Total
    LK = MQ | RESERVE

Each `LOAN` tuple indicates some sort of restriction on what can be
done to the lvalue `LV`; `LV` will always be a path owned by the
current stack frame. These restrictions are called "loans" because
they are always the result of a borrow expression.

Every loan has a lifetime `LT` during which those restrictions are in
effect.  The indicator `PT` distinguishes between *total* loans, in
which the LV itself was borrowed, and *partial* loans, which means
that some content ownwed by LV was borrowed.

The final element in the loan tuple is the *loan kind* `LK`.  There
are four kinds: mutable, immutable, const, and reserve:

- A "mutable" loan means that LV may be written to through an alias, and
  thus LV cannot be written to directly or immutably aliased (remember
  that we preserve the invariant that any given value can only be
  written to through one path at a time; hence if there is a mutable
  alias to LV, then LV cannot be written directly until this alias is
  out of scope).

- An "immutable" loan means that LV must remain immutable.  Hence it
  cannot be written, but other immutable aliases are permitted.

- A "const" loan means that an alias to LV exists.  LV may still be
  written or frozen.

- A "reserve" loan is the strongest case.  It prevents both mutation
  and aliasing of any kind, including `&const` loans.  Reserve loans
  are a side-effect of borrowing an `&mut` loan.

In addition to affecting mutability, a loan of any kind implies that
LV cannot be moved.

### Example

To give you a better feeling for what a loan is, let's look at three
loans that would be issued as a result of the borrow `&(*x).f` in the
example above:

    ((*x).f, Total, mut, 'a)
    (*x, Partial, mut, 'a)
    (x, Partial, mut, 'a)

The first loan states that the expression `(*x).f` has been loaned
totally as mutable for the lifetime `'a`. This first loan would
prevent an assignment `(*x).f = ...` from occurring during the
lifetime `'a`.

Now let's look at the second loan. You may have expected that each
borrow would result in only one loan. But this is not the case.
Instead, there will be loans for every path where mutation might
affect the validity of the borrowed pointer that is created (in some
cases, there can even be multiple loans per path, see the section on
"Borrowing in Calls" below for the gory details). The reason for this
is to prevent actions that would indirectly affect the borrowed path.
In this case, we wish to ensure that `(*x).f` is not mutated except
through the mutable alias `y`.  Therefore, we must not only prevent an
assignment to `(*x).f` but also an assignment like `*x = Foo {...}`,
as this would also mutate the field `f`.  To do so, we issue a
*partial* mutable loan for `*x` (the loan is partial because `*x`
itself was not borrowed).  This partial loan will cause any attempt to
assign to `*x` to be flagged as an error.

Because both partial and total loans prevent assignments, you may
wonder why we bother to distinguish between them.  The reason for this
distinction has to do with preventing double borrows. In particular,
it is legal to borrow both `&mut x.f` and `&mut x.g` simultaneously,
but it is not legal to borrow `&mut x.f` twice. In the borrow checker,
the first case would result in two *partial* mutable loans of `x`
(along with one total mutable loan of `x.f` and one of `x.g) whereas
the second would result in two *total* mutable loans of `x.f` (along
with two partial mutable loans of `x`).  Multiple *total mutable* loan
for the same path are not permitted, but multiple *partial* loans (of
any mutability) are permitted.

Finally, we come to the third loan. This loan is a partial mutable
loan of `x`.  This loan prevents us from reassigning `x`, which would
be bad for two reasons.  First, it would change the value of `(*x).f`
but, even worse, it would cause the pointer `y` to become a dangling
pointer.  Bad all around.

## Checking for illegal assignments, moves, and reborrows

Once we have computed the loans introduced by each borrow, the borrow
checker will determine the full set of loans in scope at each
expression and use that to decide whether that expression is legal.
Remember that the scope of loan is defined by its lifetime LT.  We
sometimes say that a loan which is in-scope at a particular point is
an "outstanding loan".

The kinds of expressions which in-scope loans can render illegal are
*assignments*, *moves*, and *borrows*.

An assignments to an lvalue LV is illegal if there is in-scope mutable
or immutable loan for LV.  Assignment with an outstanding mutable loan
is illegal because then the `&mut` pointer is supposed to be the only
way to mutate the value.  Assignment with an outstanding immutable
loan is illegal because the value is supposed to be immutable at that
point.

A move from an lvalue LV is illegal if there is any sort of
outstanding loan.

A borrow expression may be illegal if any of the loans which it
produces conflict with other outstanding loans.  Two loans are
considered compatible if one of the following conditions holds:

- At least one loan is a const loan.
- Both loans are partial loans.
- Both loans are immutable.

Any other combination of loans is illegal.

# The set of loans that results from a borrow expression

Here we'll define four functions---MUTATE, FREEZE, ALIAS, and
TAKE---which are all used to compute the set of LOANs that result
from a borrow expression.  The first three functions each have
a similar type signature:

    MUTATE(LV, LT, PT) -> LOANS
    FREEZE(LV, LT, PT) -> LOANS
    ALIAS(LV, LT, PT) -> LOANS

MUTATE, FREEZE, and ALIAS are used when computing the loans result
from mutable, immutable, and const loans respectively.  For example,
the loans resulting from an expression like `&mut (*x).f` would be
computed by `MUTATE((*x).f, LT, Total)`, where `LT` is the lifetime of
the resulting pointer.  Similarly the loans for `&(*x).f` and `&const
(*x).f` would be computed by `FREEZE((*x).f, LT, Total)` and
`ALIAS((*x).f, LT, Total)` respectively. (Actually this is a slight
simplification; see the section below on Borrows in Calls for the full
gory details)

The names MUTATE, FREEZE, and ALIAS are intended to suggest the
semantics of `&mut`, `&`, and `&const` borrows respectively.  `&mut`,
for example, creates a mutable alias of LV.  `&` causes the borrowed
value to be frozen (immutable).  `&const` does neither but does
introduce an alias to be the borrowed value.

Each of these three functions is only defined for some inputs.  That
is, it may occur that some particular borrow is not legal.  For
example, it is illegal to make an `&mut` loan of immutable data.  In
that case, the MUTATE() function is simply not defined (in the code,
it returns a Result<> condition to indicate when a loan would be
illegal).

The final function, RESERVE, is used as part of borrowing an `&mut`
pointer.  Due to the fact that it is used for one very particular
purpose, it has a rather simpler signature than the others:

    RESERVE(LV, LT) -> LOANS

It is explained when we come to that case.

## The function MUTATE()

Here we use [inference rules][ir] to define the MUTATE() function.
We will go case by case for the various kinds of lvalues that
can be borrowed.

[ir]: http://en.wikipedia.org/wiki/Rule_of_inference

### Mutating local variables

The rule for mutating local variables is as follows:

    Mutate-Variable:
      LT <= Scope(x)
      Mut(x) = Mut
      --------------------------------------------------
      MUTATE(x, LT, PT) = (x, LT, PT, mut)

Here `Scope(x)` is the lifetime of the block in which `x` was declared
and `Mut(x)` indicates the mutability with which `x` was declared.
This rule simply states that you can only create a mutable alias
to a variable if it is mutable, and that alias cannot outlive the
stack frame in which the variable is declared.

### Mutating fields and owned pointers

As it turns out, the rules for mutating fields and mutating owned
pointers turn out to be quite similar.  The reason is that the
expressions `LV.f` and `*LV` are both owned by their base expression
`LV`.  So basically the result of mutating `LV.f` or `*LV` is computed
by adding a loan for `LV.f` or `*LV` and then the loans for a partial
take of `LV`:

    Mutate-Field:
      MUTATE(LV, LT, Partial) = LOANS
      ------------------------------------------------------------
      MUTATE(LV.f, LT, PT) = LOANS, (LV.F, LT, PT, mut)

    Mutate-Owned-Ptr:
      Type(LV) = ~Ty
      MUTATE(LV, LT, Partial) = LOANS
      ------------------------------------------------------------
      MUTATE(*LV, LT, PT) = LOANS, (*LV, LT, PT, mut)

Note that while our micro-language only has fields, the slight
variations on the `Mutate-Field` rule are used for any interior content
that appears in the full Rust language, such as the contents of a
tuple, fields in a struct, or elements of a fixed-length vector.

### Mutating dereferenced borrowed pointers

The rule for borrowed pointers is by far the most complicated:

    Mutate-Mut-Borrowed-Ptr:
      Type(LV) = &LT_P mut Ty             // (1)
      LT <= LT_P                          // (2)
      RESERVE(LV, LT) = LOANS             // (3)
      ------------------------------------------------------------
      MUTATE(*LV, LT, PT) = LOANS, (*LV, LT, PT, Mut)

Condition (1) states that only a mutable borrowed pointer can be
taken.  Condition (2) states that the lifetime of the alias must be
less than the lifetime of the borrowed pointer being taken.

Conditions (3) and (4) are where things get interesting.  The intended
semantics of the borrow is that the new `&mut` pointer is the only one
which has the right to modify the data; the original `&mut` pointer
must not be used for mutation.  Because borrowed pointers do not own
their content nor inherit mutability, we must be particularly cautious
of aliases, which could permit the original borrowed pointer to be
reached from another path and thus circumvent our loans.

Here is one example of what could go wrong if we ignore clause (4):

    let x: &mut T;
    ...
    let y = &mut *x;   // Only *y should be able to mutate...
    let z = &const x;
    **z = ...;         // ...but here **z is still able to mutate!

Another possible error could occur with moves:

    let x: &mut T;
    ...
    let y = &mut *x;   // Issues loan: (*x, LT, Total, Mut)
    let z = x;         // moves from x
    *z = ...;          // Mutates *y indirectly! Bad.

In both of these cases, the problem is that when creating the alias
`y` we would only issue a loan preventing assignment through `*x`.
But this loan can be easily circumvented by moving from `x` or
aliasing it.  Note that, in the first example, the alias of `x` was
created using `&const`, which is a particularly weak form of alias.

The danger of aliases can also occur when the `&mut` pointer itself
is already located in an alias location, as here:

    let x: @mut &mut T; // or &mut &mut T, &&mut T,
    ...                 // &const &mut T, @&mut T, etc
    let y = &mut **x;   // Only *y should be able to mutate...
    let z = x;
    **z = ...;          // ...but here **z is still able to mutate!

When we cover the rules for RESERVE, we will see that it would
disallow this case, because MUTATE can only be applied to canonical
lvalues which are owned by the current stack frame.

It might be the case that if `&const` and `@const` pointers were
removed, we could do away with RESERVE and simply use MUTATE instead.
But we have to be careful about the final example in particular, since
dynamic freezing would not be sufficient to prevent this example.
Perhaps a combination of MUTATE with a predicate OWNED(LV).

One final detail: unlike every other case, when we calculate the loans
using RESERVE we do not use the original lifetime `LT` but rather
`GLB(Scope(LV), LT)`.  What this says is:

### Mutating dereferenced managed pointers

Because the correctness of managed pointer loans is checked dynamically,
the rule is quite simple:

    Mutate-Mut-Managed-Ptr:
      Type(LV) = @mut Ty
      Add ROOT-FREEZE annotation for *LV with lifetime LT
      ------------------------------------------------------------
      MUTATE(*LV, LT, Total) = []

No loans are issued.  Instead, we add a side annotation that causes
`*LV` to be rooted and frozen on entry to LV.  You could rephrase
these rules as having multiple returns values, or rephrase this as a
kind of loan, but whatever.

One interesting point is that *partial takes* of `@mut` are forbidden.
This is not for any soundness reason but just because it is clearer
for users when `@mut` values are either lent completely or not at all.

## The function FREEZE

The rules for FREEZE are pretty similar to MUTATE.  The first four
cases I'll just present without discussion, as the reasoning is
quite analogous to the MUTATE case:

    Freeze-Variable:
      LT <= Scope(x)
      --------------------------------------------------
      FREEZE(x, LT, PT) = (x, LT, PT, imm)

    Freeze-Field:
      FREEZE(LV, LT, Partial) = LOANS
      ------------------------------------------------------------
      FREEZE(LV.f, LT, PT) = LOANS, (LV.F, LT, PT, imm)

    Freeze-Owned-Ptr:
      Type(LV) = ~Ty
      FREEZE(LV, LT, Partial) = LOANS
      ------------------------------------------------------------
      FREEZE(*LV, LT, PT) = LOANS, (*LV, LT, PT, imm)

    Freeze-Mut-Borrowed-Ptr:
      Type(LV) = &LT_P mut Ty
      LT <= LT_P
      RESERVE(LV, LT) = LOANS
      ------------------------------------------------------------
      FREEZE(*LV, LT, PT) = LOANS, (*LV, LT, PT, Imm)

    Freeze-Mut-Managed-Ptr:
      Type(LV) = @mut Ty
      Add ROOT-FREEZE annotation for *LV with lifetime LT
      ------------------------------------------------------------
      Freeze(*LV, LT, Total) = []

The rule to "freeze" an immutable borrowed pointer is quite
simple, since the content is already immutable:

    Freeze-Imm-Borrowed-Ptr:
      Type(LV) = &LT_P Ty                 // (1)
      LT <= LT_P                          // (2)
      ------------------------------------------------------------
      FREEZE(*LV, LT, PT) = LOANS, (*LV, LT, PT, Mut)

The final two rules pertain to borrows of `@Ty`.  There is a bit of
subtlety here.  The main problem is that we must guarantee that the
managed box remains live for the entire borrow.  We can either do this
dynamically, by rooting it, or (better) statically, and hence there
are two rules:

    Freeze-Imm-Managed-Ptr-1:
      Type(LV) = @Ty
      Add ROOT annotation for *LV
      ------------------------------------------------------------
      FREEZE(*LV, LT, PT) = []

    Freeze-Imm-Managed-Ptr-2:
      Type(LV) = @Ty
      LT <= Scope(LV)
      Mut(LV) = imm
      LV is not moved
      ------------------------------------------------------------
      FREEZE(*LV, LT, PT) = []

The intention of the second rule is to avoid an extra root if LV
serves as a root.  In that case, LV must (1) outlive the borrow; (2)
be immutable; and (3) not be moved.

## The ALIAS function

The function ALIAS is used for `&const` loans but also to handle one
corner case concerning function arguments (covered in the section
"Borrows in Calls" below).  It computes the loans that result from
observing that there is a pointer to `LV` and thus that pointer must
remain valid.

The first two rules are simple:

    Alias-Variable:
      LT <= Scope(x)
      --------------------------------------------------
      ALIAS(x, LT, PT) = (x, LT, PT, Const)

    Alias-Field:
      ALIAS(LV, LT, Partial) = LOANS
      ------------------------------------------------------------
      ALIAS(LV.f, LT, PT) = LOANS, (LV.F, LT, PT, Const)

### Aliasing owned pointers

The rule for owned pointers is somewhat interesting:

    Alias-Owned-Ptr:
      Type(LV) = ~Ty
      FREEZE(LV, LT, Partial) = LOANS
      ------------------------------------------------------------
      ALIAS(*LV, LT, PT) = LOANS, (*LV, LT, PT, Const)

Here we *freeze* the base `LV`.  The reason is that if an owned
pointer is mutated it frees its content, which means that the alias to
`*LV` would become a dangling pointer.

### Aliasing borrowed pointers

The rule for borrowed pointers is quite simple, because borrowed
pointers do not own their content and thus do not play a role in
keeping it live:

    Alias-Borrowed-Ptr:
      Type(LV) = &LT_P MQ Ty
      LT <= LT_P
      ------------------------------------------------------------
      ALIAS(*LV, LT, PT) = []

Basically, the existence of a borrowed pointer to some memory with
lifetime LT_P is proof that the memory can safely be aliased for any
lifetime LT <= LT_P.

### Aliasing managed pointers

The rules for aliasing managed pointers are similar to those
used with FREEZE, except that they apply to all manager pointers
regardles of mutability:

    Alias-Managed-Ptr-1:
      Type(LV) = @MQ Ty
      Add ROOT annotation for *LV
      ------------------------------------------------------------
      ALIAS(*LV, LT, PT) = []

    Alias-Managed-Ptr-2:
      Type(LV) = @MQ Ty
      LT <= Scope(LV)
      Mut(LV) = imm
      LV is not moved
      ------------------------------------------------------------
      ALIAS(*LV, LT, PT) = []

## The RESERVE function

The final function, RESERVE, is used for loans of `&mut` pointers.  As
discussed in the section on the function MUTATE, we must be quite
careful when "re-borrowing" an `&mut` pointer to ensure that the original
`&mut` pointer can no longer be used to mutate.

There are a couple of dangers to be aware of:

- `&mut` pointers do not inherit mutability.  Therefore, if you have
  an lvalue LV with type `&mut T` and you freeze `LV`, you do *not*
  freeze `*LV`.  This is quite different from an `LV` with type `~T`.

- Also, because they do not inherit mutability, if the `&mut` pointer
  lives in an aliased location, then *any alias* can be used to write!

As a consequence of these two rules, RESERVE can only be successfully
invoked on an lvalue LV that is *owned by the current stack frame*.
This ensures that there are no aliases that are not visible from the
outside.  Moreover, Reserve loans are incompatible with all other
loans, even Const loans.  This prevents any aliases from being created
within the current function.

### Reserving local variables

The rule for reserving a variable is generally straightforward but
with one interesting twist:

    Reserve-Variable:
      --------------------------------------------------
      RESERVE(x, LT) = (x, LT, Total, Reserve)

The twist here is that the incoming lifetime is not required to
be a subset of the incoming variable, unlike every other case.  To
see the reason for this, imagine the following function:

    struct Foo { count: uint }
    fn count_field(x: &'a mut Foo) -> &'a mut count {
        &mut (*x).count
    }

This function consumes one `&mut` pointer and returns another with the
same lifetime pointing at a particular field.  The borrow for the
`&mut` expression will result in a call to `RESERVE(x, 'a)`, which is
intended to guarantee that `*x` is not later aliased or used to
mutate.  But the lifetime of `x` is limited to the current function,
which is a sublifetime of the parameter `'a`, so the rules used for
MUTATE, FREEZE, and ALIAS (which require that the lifetime of the loan
not exceed the lifetime of the variable) would result in an error.

Nonetheless this function is perfectly legitimate.  After all, the
caller has moved in an `&mut` pointer with lifetime `'a`, and thus has
given up their right to mutate the value for the remainder of `'a`.
So it is fine for us to return a pointer with the same lifetime.

The reason that RESERVE differs from the other functions is that
RESERVE is not responsible for guaranteeing that the pointed-to data
will outlive the borrowed pointer being created.  After all, `&mut`
values do not own the data they point at.

### Reserving owned content

The rules for fields and owned pointers are very straightforward:

    Reserve-Field:
      RESERVE(LV, LT) = LOANS
      ------------------------------------------------------------
      RESERVE(LV.f, LT) = LOANS, (LV.F, LT, Total, Reserve)

    Reserve-Owned-Ptr:
      Type(LV) = ~Ty
      RESERVE(LV, LT) = LOANS
      ------------------------------------------------------------
      RESERVE(*LV, LT) = LOANS, (*LV, LT, Total, Reserve)

### Reserving `&mut` borrowed pointers

Unlike other borrowed pointers, `&mut` pointers are unaliasable,
so we can reserve them like everything else:

    Reserve-Mut-Borrowed-Ptr:
      Type(LV) = &LT_P mut Ty
      RESERVE(LV, LT) = LOANS
      ------------------------------------------------------------
      RESERVE(*LV, LT) = LOANS, (*LV, LT, Total, Reserve)

## Borrows in calls

Earlier we said that the MUTATE, FREEZE, and ALIAS functions were used
to compute the loans resulting from a borrow expression.  But this is
not strictly correct, there is a slight complication that occurs with
calls by which additional loans may be necessary.  We will explain
that here and give the full details.

Imagine a call expression `'a: E1(E2, E3)`, where `Ei` are some
expressions. If we break this down to something a bit lower-level, it
is kind of short for:

    'a: {
        'a_arg1: let temp1: ... = E1;
        'a_arg2: let temp2: ... = E2;
        'a_arg3: let temp3: ... = E3;
        'a_call: temp1(temp2, temp3)
    }

Here the lifetime labels indicate the various lifetimes. As you can
see there are in fact four relevant lifetimes (only one of which was
named by the user): `'a` corresponds to the expression `E1(E2, E3)` as
a whole. `'a_arg1`, `'a_arg2`, and `'a_arg3` correspond to the
evaluations of `E1`, `E2`, and `E3` respectively. Finally, `'a_call`
corresponds to the *actual call*, which is the point where the values
of the parameters will be used.

Now, let's look at a (contrived, but representative) example to see
why all this matters:

    struct Foo { f: uint, g: uint }
    ...
    fn add(p: &mut uint, v: uint) {
        *p += v;
    }
    ...
    fn inc(p: &mut uint) -> uint {
        *p += 1; *p
    }
    fn weird() {
        let mut x: ~Foo = ~Foo { ... };
        'a: add(&mut (*x).f,
                'b: inc(&mut (*x).f)) // (*)
    }

The important part is the line marked `(*)` which contains a call to
`add()`. The first argument is a mutable borrow of the field `f`.
The second argument *always borrows* the field `f`. Now, if these two
borrows overlapped in time, this would be illegal, because there would
be two `&mut` pointers pointing at `f`. And, in a way, they *do*
overlap in time, since the first argument will be evaluated first,
meaning that the pointer will exist when the second argument executes.
But in another important way they do not overlap in time. Let's
expand out that final call to `add()` as we did before:

    'a: {
        'a_arg1: let a_temp1: ... = add;
        'a_arg2: let a_temp2: &'a_call mut uint = &'a_call mut (*x).f;
        'a_arg3_: let a_temp3: uint = {
            let b_temp1: ... = inc;
            let b_temp2: &'b_call = &'b_call mut (*x).f;
            'b_call: b_temp1(b_temp2)
        };
        'a_call: a_temp1(a_temp2, a_temp3)
    }

When it's written this way, we can see that although there are two
borrows, the first has lifetime `'a_call` and the second has lifetime
`'b_call` and in fact these lifetimes do not overlap. So everything
is fine.

But this does not mean that there isn't reason for caution!  Imagine a
devious program like *this* one:

    struct Foo { f: uint, g: uint }
    ...
    fn add(p: &mut uint, v: uint) {
        *p += v;
    }
    ...
    fn consume(x: ~Foo) -> uint {
        x.f + x.g
    }
    fn weird() {
        let mut x: ~Foo = ~Foo { ... };
        'a: add(&mut (*x).f, consume(x)) // (*)
    }

In this case, there is only one borrow, but the second argument is
`consume(x)` instead of a second borrow. Because `consume()` is
declared to take a `~Foo`, it will in fact free the pointer `x` when
it has finished executing. If it is not obvious why this is
troublesome, consider this expanded version of that call:

    'a: {
        'a_arg1: let a_temp1: ... = add;
        'a_arg2: let a_temp2: &'a_call mut uint = &'a_call mut (*x).f;
        'a_arg3_: let a_temp3: uint = {
            let b_temp1: ... = consume;
            let b_temp2: ~Foo = x;
            'b_call: b_temp1(x)
        };
        'a_call: a_temp1(a_temp2, a_temp3)
    }

In this example, we will have borrowed the first argument before `x`
is freed and then free `x` during evaluation of the second
argument. This causes `a_temp2` to be invalidated.

Of course the loans computed from the borrow expression are supposed
to prevent this situation.  But if we just considered the loans from
`MUTATE((*x).f, 'a_call, Total)`, the resulting loans would be:

    ((*x).f, 'a_call, Total,   Mut)
    (*x,     'a_call, Partial, Mut)
    (x,      'a_call, Partial, Mut)

Because these loans are only in scope for `'a_call`, they do nothing
to prevent the move that occurs evaluating the second argument.

The way that we solve this is to say that if you have a borrow
expression `&'LT_P mut LV` which itself occurs in the lifetime
`'LT_B`, then the resulting loans are:

    MUTATE(LV, LT_P, Total) + ALIAS(LV, LUB(LT_P, LT_B), Total)

The call to MUTATE is what we've seen so far.  The second part
expresses the idea that the expression LV will be evaluated starting
at LT_B until the end of LT_P.  Now, in the normal case, LT_P >= LT_B,
and so the second set of loans that result from a ALIAS are basically
a no-op.  However, in the case of an argument where the evaluation of
the borrow occurs before the interval where the resulting pointer will
be used, this ALIAS is important.

In the case of our example, it would produce a set of loans like:

    ((*x).f, 'a, Total, Const)
    (*x, 'a, Total, Const)
    (x, 'a, Total, Imm)

The scope of these loans is `'a = LUB('a_arg2, 'a_call)`, and so they
encompass all subsequent arguments.  The first set of loans are Const
loans, which basically just prevent moves.  However, when we cross
over the dereference of the owned pointer `x`, the rule for ALIAS
specifies that `x` must be frozen, and hence the final loan is an Imm
loan.  In any case the troublesome second argument would be flagged
as an error.

# Maps that are created

Borrowck results in two maps.

- `root_map`: identifies those expressions or patterns whose result
  needs to be rooted. Conceptually the root_map maps from an
  expression or pattern node to a `node_id` identifying the scope for
  which the expression must be rooted (this `node_id` should identify
  a block or call). The actual key to the map is not an expression id,
  however, but a `root_map_key`, which combines an expression id with a
  deref count and is used to cope with auto-deref.

*/
