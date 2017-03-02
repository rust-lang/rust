- Start Date: 2014-07-29
- RFC PR: [rust-lang/rfcs#130](https://github.com/rust-lang/rfcs/pull/130)
- Rust Issue: [rust-lang/rust#16094](https://github.com/rust-lang/rust/issues/16094)

# Summary

Remove special treatment of `Box<T>` from the borrow checker.

# Motivation

Currently the `Box<T>` type is special-cased and converted to the old
`~T` internally. This is mostly invisible to the user, but it shows up
in some places that give special treatment to `Box<T>`. This RFC is
specifically concerned with the fact that the borrow checker has
greater precision when derefencing `Box<T>` vs other smart pointers
that rely on the `Deref` traits. Unlike the other kinds of special
treatment, we do not currently have a plan for how to extend this
behavior to all smart pointer types, and hence we would like to remove
it.

Here is an example that illustrates the extra precision afforded to
`Box<T>` vs other types that implement the `Deref` traits. The
following program, written using the `Box` type, compiles
successfully:

    struct Pair {
        a: uint,
        b: uint
    }
    
    fn example1(mut smaht: Box<Pair>) {
        let a = &mut smaht.a;
        let b = &mut smaht.b;
        ...
    }

This program compiles because the type checker can see that
`(*smaht).a` and `(*smaht).b` are always distinct paths. In contrast,
if I use a smart pointer, I get compilation errors:

    fn example2(cell: RefCell<Pair>) {
        let mut smaht: RefMut<Pair> = cell.borrow_mut();
        let a = &mut smaht.a;
        
        // Error: cannot borrow `smaht` as mutable more than once at a time
        let b = &mut smaht.b;
    }

To see why this, consider the desugaring:

    fn example2(smaht: RefCell<Pair>) {
        let mut smaht = smaht.borrow_mut();
        
        let tmp1: &mut Pair = smaht.deref_mut(); // borrows `smaht`
        let a = &mut tmp1.a;
        
        let tmp2: &mut Pair = smaht.deref_mut(); // borrows `smaht` again!
        let b = &mut tmp2.b;
    }

It is a violation of the Rust type system to invoke `deref_mut` when
the reference to `a` is valid and usable, since `deref_mut` requires
`&mut self`, which in turn implies no alias to `self` or anything
owned by `self`.

This desugaring suggests how the problem can be worked around in user
code. The idea is to pull the result of the deref into a new temporary:

    fn example3(smaht: RefCell<Pair>) {
        let mut smaht: RefMut<Pair> = smaht.borrow_mut();
        let temp: &mut Pair = &mut *smaht;
        let a = &mut temp.a;
        let b = &mut temp.b;
    }

# Detailed design

Removing this treatment from the borrow checker basically means
changing the construction of loan paths for unique pointers.

I don't actually know how best to implement this in the borrow
checker, particularly concerning the desire to keep the ability to
move out of boxes and use them in patterns. This requires some
investigation. The easiest and best way may be to "do it right" and is
probably to handle derefs of `Box<T>` in a similar way to how
overloaded derefs are handled, but somewhat differently to account for
the possibility of moving out of them. Some investigation is needed.

# Drawbacks

The borrow checker rules are that much more restrictive.

# Alternatives

We have ruled out inconsistent behavior between `Box` and other smart
pointer types. We considered a number of ways to extend the current
treatment of box to other smart pointer types:

1. *Require* compiler to introduce deref temporaries automatically
   where possible. This is plausible as a future extension but
   requires some thought to work through all cases. It may be
   surprising. Note that this would be a required optimization because
   if the optimization is not performed it affects what programs can
   successfully type check. (Naturally it is also observable.)
   
2. Some sort of unsafe deref trait that acknolwedges possibliity of
   other pointers into the referent. Unappealing because the problem
   is not that bad as to require unsafety.
   
3. Determining conditions (perhaps based on parametricity?) where it
   is provably safe to call deref. It is dubious and unknown if such
   conditions exist or what that even means. Rust also does not really
   enjoy parametricity properties due to presence of reflection and
   unsafe code.

# Unresolved questions

Best implementation strategy.
