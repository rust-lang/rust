# Argument passing

Rust datatypes are not trivial to copy (the way, for example,
JavaScript values can be copied by simply taking one or two machine
words and plunking them somewhere else). Shared boxes require
reference count updates, big records or tags require an arbitrary
amount of data to be copied (plus updating the reference counts of
shared boxes hanging off them), unique pointers require their origin
to be de-initialized.

For this reason, the way Rust passes arguments to functions is a bit
more involved than it is in most languages. It performs some
compile-time cleverness to get rid of most of the cost of copying
arguments, and forces you to put in explicit copy operators in the
places where it can not.

## Safe references

The foundation of Rust's argument-passing optimization is the fact
that Rust tasks for single-threaded worlds, which share no data with
other tasks, and that most data is immutable.

Take the following program:

    let x = get_really_big_record();
    myfunc(x);

We want to pass `x` to `myfunc` by pointer (which is easy), *and* we
want to ensure that `x` stays intact for the duration of the call
(which, in this example, is also easy). So we can just use the
existing value as the argument, without copying.

There are more involved cases. The call could look like this:

    myfunc(x, {|| x = get_another_record(); });

Now, if `myfunc` first calls its second argument and then accesses its
first argument, it will see a different value from the one that was
passed to it.

The compiler will insert an implicit copy of `x` in such a case,
*except* if `x` contains something mutable, in which case a copy would
result in code that behaves differently (if you mutate the copy, `x`
stays unchanged). That would be bad, so the compiler will disallow
such code.

When inserting an implicit copy for something big, the compiler will
warn, so that you know that the code is not as efficient as it looks.

There are even more tricky cases, in which the Rust compiler is forced
to pessimistically assume a value will get mutated, even though it is
not sure.

    fn for_each(v: [mutable @int], iter: block(@int)) {
       for elt in v { iter(elt); }
    }

For all this function knows, calling `iter` (which is a closure that
might have access to the vector that's passed as `v`) could cause the
elements in the vector to be mutated, with the effect that it can not
guarantee that the boxes will live for the duration of the call. So it
has to copy them. In this case, this will happen implicitly (bumping a
reference count is considered cheap enough to not warn about it).

## The copy operator

If the `for_each` function given above were to take a vector of
`{mutable a: int}` instead of `@int`, it would not be able to
implicitly copy, since if the `iter` function changes a copy of a
mutable record, the changes won't be visible in the record itself. If
we *do* want to allow copies there, we have to explicitly allow it
with the `copy` operator:

    type mutrec = {mutable x: int};
    fn for_each(v: [mutable mutrec], iter: block(mutrec)) {
       for elt in v { iter(copy elt); }
    }

## Argument passing styles

The fact that arguments are conceptually passed by safe reference does
not mean all arguments are passed by pointer. Composite types like
records and tags *are* passed by pointer, but others, like integers
and pointers, are simply passed by value.

It is possible, when defining a function, to specify a passing style
for a parameter by prefixing the parameter name with a symbol. The
most common special style is by-mutable-reference, written `&`:

    fn vec_push(&v: [int], elt: int) {
        v += [elt];
    }

This will make it possible for the function to mutate the parameter.
Clearly, you are only allowed to pass things that can actually be
mutated to such a function.

Another style is by-move, which will cause the argument to become
de-initialized on the caller side, and give ownership of it to the
called function. This is written `-`.

Finally, the default passing styles (by-value for non-structural
types, by-reference for structural ones) are written `+` for by-value
and `&&` for by(-immutable)-reference. It is sometimes necessary to
override the defaults. We'll talk more about this when discussing
[generics][gens].

[gens]: FIXME

## Other uses of safe references

Safe references are not only used for argument passing. When you
destructure on a value in an `alt` expression, or loop over a vector
with `for`, variables bound to the inside of the given data structure
will use safe references, not copies. This means such references have
little overhead, but you'll occasionally have to copy them to ensure
safety.

    let my_rec = {a: 4, b: [1, 2, 3]};
    alt my_rec {
      {a, b} {
        log b; // This is okay
        my_rec = {a: a + 1, b: b + [a]};
        log b; // Here reference b has become invalid
      }
    }
