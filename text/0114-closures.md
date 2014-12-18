- Start Date: 2014-07-29
- RFC PR: [rust-lang/rfcs#114](https://github.com/rust-lang/rfcs/pull/114)
- Rust Issue: [rust-lang/rust#16095](https://github.com/rust-lang/rust/issues/16095)

# Summary

- Convert function call `a(b, ..., z)` into an overloadable operator
  via the traits `Fn<A,R>`, `FnShare<A,R>`, and `FnOnce<A,R>`, where `A`
  is a tuple `(B, ..., Z)` of the types `B...Z` of the arguments
  `b...z`, and `R` is the return type. The three traits differ in
  their self argument (`&mut self` vs `&self` vs `self`).
- Remove the `proc` expression form and type.
- Remove the closure types (though the form lives on as syntactic
  sugar, see below).
- Modify closure expressions to permit specifying by-reference vs
  by-value capture and the receiver type:
  - Specifying by-reference vs by-value closures:
    - `ref |...| expr` indicates a closure that captures upvars from the
      environment by reference. This is what closures do today and the
      behavior will remain unchanged, other than requiring an explicit
      keyword.
    - `|...| expr` will therefore indicate a closure that captures upvars
      from the environment by value. As usual, this is either a copy or
      move depending on whether the type of the upvar implements `Copy`.
  - Specifying receiver mode (orthogonal to capture mode above):
    - `|a, b, c| expr` is equivalent to `|&mut: a, b, c| expr`
    - `|&mut: ...| expr` indicates that the closure implements `Fn`
    - `|&: ...| expr` indicates that the closure implements `FnShare`
    - `|: a, b, c| expr` indicates that the closure implements `FnOnce`.
- Add syntactic sugar where `|T1, T2| -> R1` is translated to
  a reference to one of the fn traits as follows:
  - `|T1, ..., Tn| -> R` is translated to `Fn<(T1, ..., Tn), R>`
  - `|&mut: T1, ..., Tn| -> R` is translated to `Fn<(T1, ..., Tn), R>`
  - `|&: T1, ..., Tn| -> R` is translated to `FnShare<(T1, ..., Tn), R>`
  - `|: T1, ..., Tn| -> R` is translated to `FnOnce<(T1, ..., Tn), R>`
  
One aspect of closures that this RFC does *not* describe is that we
must permit trait references to be universally quantified over regions
as closures are today. A description of this change is described below
under *Unresolved questions* and the details will come in a
forthcoming RFC.

# Motivation

Over time we have observed a very large number of possible use cases
for closures. The goal of this RFC is to create a unified closure
model that encompasses all of these use cases.

Specific goals (explained in more detail below):

1. Give control over inlining to users.
2. Support closures that bind by reference and closures that bind by value.
3. Support different means of accessing the closure environment,
   corresponding to `self`, `&self`, and `&mut self` methods.
   
As a side benefit, though not a direct goal, the RFC reduces the
size/complexity of the language's core type system by unifying
closures and traits.

## The core idea: unifying closures and traits

The core idea of the RFC is to unify closures, procs, and
traits. There are a number of reasons to do this. First, it simplifies
the language, because closures, procs, and traits already served
similar roles and there was sometimes a lack of clarity about which
would be the appropriate choice. However, in addition, the unification
offers increased expressiveness and power, because traits are a more
generic model that gives users more control over optimization.

The basic idea is that function calls become an overridable operator.
Therefore, an expression like `a(...)` will be desugar into an
invocation of one of the following traits:

    trait Fn<A,R> {
        fn call(&mut self, args: A) -> R;
    }

    trait FnShare<A,R> {
        fn call_share(&self, args: A) -> R;
    }

    trait FnOnce<A,R> {
        fn call_once(self, args: A) -> R;
    }

Essentially, `a(b, c, d)` becomes sugar for one of the following:

    Fn::call(&mut a, (b, c, d))
    FnShare::call_share(&a, (b, c, d))
    FnOnce::call_once(a, (b, c, d))

To integrate with this, closure expressions are then translated into a
fresh struct that implements one of those three traits. The precise
trait is currently indicated using explicit syntax but may eventually
be inferred.

This change gives user control over virtual vs static dispatch.  This
works in the same way as generic types today:

    fn foo(x: &mut Fn<(int,),int>) -> int {
        x(2) // virtual dispatch
    }

    fn foo<F:Fn<(int,),int>>(x: &mut F) -> int {
        x(2) // static dispatch
    }

The change also permits returning closures, which is not currently
possible (the example relies on the proposed `impl` syntax from
rust-lang/rfcs#105):

    fn foo(x: impl Fn<(int,),int>) -> impl Fn<(int,),int> {
        |v| x(v * 2)
    }
    
Basically, in this design there is nothing special about a closure.
Closure expressions are simply a convenient way to generate a struct
that implements a suitable `Fn` trait.

## Bind by reference vs bind by value

When creating a closure, it is now possible to specify whether the
closure should capture variables from its environment ("upvars") by
reference or by value. The distinction is indicated using the leading
keyword `ref`:

    || foo(a, b)      // captures `a` and `b` by value
    
    ref || foo(a, b)  // captures `a` and `b` by reference, as today

### Reasons to bind by value

Bind by value is useful when creating closures that will escape from
the stack frame that created them, such as task bodies (`spawn(||
...)`) or combinators. It is also useful for moving values out of a
closure, though it should be possible to enable that with bind by
reference as well in the future.

### Reasons to bind by reference

Bind by reference is useful for any case where the closure is known
not to escape the creating stack frame. This frequently occurs
when using closures to encapsulate common control-flow patterns:

    map.insert_or_update_with(key, value, || ...)
    opt_val.unwrap_or_else(|| ...)
    
In such cases, the closure frequently wishes to read or modify local
variables on the enclosing stack frame. Generally speaking, then, such
closures should capture variables by-reference -- that is, they should
store a reference to the variable in the creating stack frame, rather
than copying the value out. Using a reference allows the closure to
mutate the variables in place and also avoids moving values that are
simply read temporarily.

The vast majority of closures in use today are should be "by
reference" closures. The only exceptions are those closures that wish
to "move out" from an upvar (where we commonly use the so-called
"option dance" today). In fact, even those closures could be "by
reference" closures, but we will have to extend the inference to
selectively identify those variables that must be moved and take those
"by value".

# Detailed design

## Closure expression syntax

Closure expressions will have the following form (using EBNF notation,
where `[]` denotes optional things and `{}` denotes a comma-separated
list):

    CLOSURE = ['ref'] '|' [SELF] {ARG} '|' ['->' TYPE] EXPR
    SELF    =  ':' | '&' ':' | '&' 'mut' ':'
    ARG     = ID [ ':' TYPE ]

The optional keyword `ref` is used to indicate whether this closure
captures *by reference* or *by value*.

Closures are always translated into a fresh struct type with one field
per upvar. In a by-value closure, the types of these fields will be
the same as the types of the corresponding upvars (modulo `&mut`
reborrows, see below). In a by-reference closure, the types of these
fields will be a suitable reference (`&`, `&mut`, etc) to the
variables being borrowed.

### By-value closures

The default form for a closure is by-value. This implies that all
upvars which are referenced are copied/moved into the closure as
appropriate. There is one special case: if the type of the value to be
moved is `&mut`, we will "reborrow" the value when it is copied into
the closure. That is, given an upvar `x` of type `&'a mut T`, the
value which is actually captured will have type `&'b mut T` where `'b
<= 'a`. This rule is consistent with our general treatment of `&mut`,
which is to aggressively reborrow wherever possible; moreover, this
rule cannot introduce additional compilation errors, it can only make
more programs successfully typecheck.

### By-reference closures

A *by-reference* closure is a convenience form in which values used in
the closure are converted into references before being captured. 
By-reference closures are always rewritable into by-value closures if
desired, but the rewrite can often be cumbersome and annoying.

Here is a (rather artificial) example of a by-reference closure in
use:

    let in_vec: Vec<int> = ...;
    let mut out_vec: Vec<int> = Vec::new();
    let opt_int: Option<int> = ...;
    
    opt_int.map(ref |v| {
        out_vec.push(v);
        in_vec.fold(v, |a, &b| a + b)
    });

This could be rewritten into a by-value closure as follows:

    let in_vec: Vec<int> = ...;
    let mut out_vec: Vec<int> = Vec::new();
    let opt_int: Option<int> = ...;

    opt_int.map({
        let in_vec = &in_vec;
        let out_vec = &mut in_vec;
        |v| {
            out_vec.push(v);
            in_vec.fold(v, |a, &b| a + b)
        }
    })
    
In this case, the capture closed over two variables, `in_vec` and
`out_vec`. As you can see, the compiler automatically infers, for each
variable, how it should be borrowed and inserts the appropriate
capture.

In the body of a `ref` closure, the upvars continue to have the same
type as they did in the outer environment. For example, the type of a
reference to `in_vec` in the above example is always `Vec<int>`,
whether or not it appears as part of a `ref` closure. This is not only
convenient, it is required to make it possible to infer whether each
variable is borrowed as an `&T` or `&mut T` borrow.

Note that there are some cases where the compiler internally employs a
form of borrow that is not available in the core language,
`&uniq`. This borrow does not permit aliasing (like `&mut`) but does
not require mutability (like `&`). This is required to allow
transparent closing over of `&mut` pointers as
[described in this blog post][p].
    
**Evolutionary note:** It is possible to evolve by-reference
closures in the future in a backwards compatible way. The goal would
be to cause more programs to type-check by default. Two possible
extensions follow:

- Detect when values are *moved* and hence should be taken by value
  rather than by reference. (This is only applicable to once
  closures.)
- Detect when it is only necessary to borrow a sub-path. Imagine a
  closure like `ref || use(&context.variable_map)`. Currently, this
  closure will borrow `context`, even though it only *uses* the field
  `variable_map`. As a result, it is sometimes necessary to rewrite
  the closure to have the form `{let v = &context.variable_map; ||
  use(v)}`.  In the future, however, we could extend the inference so
  that rather than borrowing `context` to create the closure, we would
  borrow `context.variable_map` directly.

## Closure sugar in trait references

The current type for closures, `|T1, T2| -> R`, will be repurposed as
syntactic sugar for a reference to the appropriate `Fn` trait. This
shorthand be used any place that a trait reference is appropriate. The
full type will be written as one of the following:

    <'a...'z> |T1...Tn|: K -> R
    <'a...'z> |&mut: T1...Tn|: K -> R
    <'a...'z> |&: T1...Tn|: K -> R
    <'a...'z> |: T1...Tn|: K -> R
    
Each of which would then be translated into the following trait
references, respectively:

    <'a...'z> Fn<(T1...Tn), R> + K
    <'a...'z> Fn<(T1...Tn), R> + K
    <'a...'z> FnShare<(T1...Tn), R> + K
    <'a...'z> FnOnce<(T1...Tn), R> + K

Note that the bound lifetimes `'a...'z` are not in scope for the bound
`K`.

# Drawbacks

This model is more complex than the existing model in some respects
(but the existing model does not serve the full set of desired use cases).

# Alternatives

There is one aspect of the design that is still under active
discussion:

**Introduce a more generic sugar.** It was proposed that we could
introduce `Trait(A, B) -> C` as syntactic sugar for `Trait<(A,B),C>`
rather than retaining the form `|A,B| -> C`. This is appealing but
removes the correspondence between the expression form and the
corresponding type. One (somewhat open) question is whether there will
be additional traits that mirror fn types that might benefit from this
more general sugar.

**Tweak trait names.** In conjunction with the above, there is some
concern that the type name `fn(A) -> B` for a bare function with no
environment is too similar to `Fn(A) -> B` for a closure.  To remedy
that, we could change the name of the trait to something like
`Closure(A) -> B` (naturally the other traits would be renamed to
match).

Then there are a large number of permutations and options that were
largely rejected:

**Only offer by-value closures.** We tried this and found it
required a lot of painful rewrites of perfectly reasonable code.

**Make by-reference closures the default.** We felt this was
inconsistent with the language as a whole, which tends to make "by
value" the default (e.g., `x` vs `ref x` in patterns, `x` vs `&x` in
expressions, etc.).

**Use a capture clause syntax that borrows individual variables.** "By
value" closures combined with `let` statements already serve this
role. Simply specifying "by-reference closure" also gives us room to
continue improving inference in the future in a backwards compatible
way. Moreover, the syntactic space around closures expressions is
extremely constrained and we were unable to find a satisfactory
syntax, particularly when combined with self-type annotations.
Finally, if we decide we *do* want the ability to have "mostly
by-value" closures, we can easily extend the current syntax by writing
something like `(ref x, ref mut y) || ...` etc.

**Retain the proc expression form.** It was proposed that we could
retain the `proc` expression form to specify a by-value closure and
have `||` expressions be by-reference. Frankly, the main objection to
this is that nobody likes the `proc` keyword.

**Use variadic generics in place of tuple arguments.** While variadic
generics are an interesting addition in their own right, we'd prefer
not to introduce a dependency between closures and variadic
generics. Having all arguments be placed into a tuple is also a
simpler model overall. Moreover, native ABIs on platforms of interest
treat a structure passed by value identically to distinct
arguments. Finally, given that trait calls have the "Rust" ABI, which
is not specified, we can always tweak the rules if necessary (though
there are advantages for tooling when the Rust ABI closely matches the
native ABI).

**Use inference to determine the self type of a closure rather than an
annotation.** We retain this option for future expansion, but it is
not clear whether we can always infer the self type of a
closure. Moreover, using inference rather a default raises the
question of what to do for a type like `|int| -> uint`, where
inference is not possible.

**Default to something other than `&mut self`.** It is our belief that
this is the most common use case for closures.

# Transition plan

TBD. pcwalton is working furiously as we speak.

# Unresolved questions

**What relationship should there be between the closure
traits?** On the one hand, there is clearly a relationship between the
traits.  For example, given a `FnShare`, one can easily implement
`Fn`:

    impl<A,R,T:FnShare<A,R>> Fn<A,R> for T {
        fn call(&mut self, args: A) -> R {
            (&*self).call_share(args)
        }
    }

Similarly, given a `Fn` or `FnShare`, you can implement `FnOnce`. From
this, one might derive a subtrait relationship:

    trait FnOnce { ... }
    trait Fn : FnOnce { ... }
    trait FnShare : Fn { ... }

Employing this relationship, however, would require that any manual
implement of `FnShare` or `Fn` must implement adapters for the other
two traits, since a subtrait cannot provide a specialized default of
supertrait methods (yet?). On the other hand, having no relationship
between the traits limits reuse, at least without employing explicit
adapters.

Other alternatives that have been proposed to address the problem:

- Use impls to implement the fn traits in terms of one another,
  similar to what is shown above. The problem is that we would need to
  implement `FnOnce` both for all `T` where `T:Fn` and for all `T`
  where `T:FnShare`. This will yield coherence errors unless we extend
  the language with a means to declare traits as mutually exclusive
  (which might be valuable, but no such system has currently been
  proposed nor agreed upon).

- Have the compiler implement multiple traits for a single closure.
  As with supertraits, this would require manual implements to
  implement multiple traits. It would also require generic users to
  write `T:Fn+FnMut` or else employ an explicit adapter. On the other
  hand, it preserves the "one method per trait" rule described below.

**Can we optimize away the trait vtable?** The runtime representation
of a reference `&Trait` to a trait object (and hence, under this
proposal, closures as well) is a pair of pointers `(data, vtable)`. It
has been proposed that we might be able to optimize this
representation to `(data, fnptr)` so long as `Trait` has a single
function. This slightly improves the performance of invoking the
function as one need not indirect through the vtable. The actual
implications of this on performance are unclear, but it might be a
reason to keep the closure traits to a single method.

## Closures that are quantified over lifetimes

A separate RFC is needed to describe bound lifetimes in trait
references. For example, today one can write a type like `<'a> |&'a A|
-> &'a B`, which indicates a closure that takes and returns a
reference with the same lifetime specified by the caller at each
call-site. Note that a trait reference like `Fn<(&'a A), &'a B>`,
while syntactically similar, does *not* have the same meaning because
it lacks the universal quantifier `<'a>`. Therefore, in the second
case, `'a` refers to some specific lifetime `'a`, rather than being a
lifetime parameter that is specified at each callsite. The high-level
summary of the change therefore is to permit trait references like
`<'a> Fn<(&'a A), &'a B>`; in this case, the value of `<'a>` will be
specified each time a method or other member of the trait is accessed.

[p]: http://smallcultfollowing.com/babysteps/blog/2014/05/13/focusing-on-ownership/
