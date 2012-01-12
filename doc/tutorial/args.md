# Argument passing

Rust datatypes are not trivial to copy (the way, for example,
JavaScript values can be copied by simply taking one or two machine
words and plunking them somewhere else). Shared boxes require
reference count updates, big records, tags, or unique pointers require
an arbitrary amount of data to be copied (plus updating the reference
counts of shared boxes hanging off them).

For this reason, the default calling convention for Rust functions
leaves ownership of the arguments with the caller. The caller
guarantees that the arguments will outlive the call, the callee merely
gets access to them.

## Safe references

There is one catch with this approach: sometimes the compiler can
*not* statically guarantee that the argument value at the caller side
will survive to the end of the call. Another argument might indirectly
refer to it and be used to overwrite it, or a closure might assign a
new value to it.

Fortunately, Rust tasks are single-threaded worlds, which share no
data with other tasks, and that most data is immutable. This allows
most argument-passing situations to be proved safe without further
difficulty.

Take the following program:

    # fn get_really_big_record() -> int { 1 }
    # fn myfunc(a: int) {}
    fn main() {
        let x = get_really_big_record();
        myfunc(x);
    }

Here we know for sure that no one else has access to the `x` variable
in `main`, so we're good. But the call could also look like this:

    # fn myfunc(a: int, b: block()) {}
    # fn get_another_record() -> int { 1 }
    # let x = 1;
    myfunc(x, {|| x = get_another_record(); });

Now, if `myfunc` first calls its second argument and then accesses its
first argument, it will see a different value from the one that was
passed to it.

In such a case, the compiler will insert an implicit copy of `x`,
*except* if `x` contains something mutable, in which case a copy would
result in code that behaves differently. If copying `x` might be
expensive (for example, if it holds a vector), the compiler will emit
a warning.

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

Adding a `copy` operator is also the way to muffle warnings about
implicit copies.

## Other uses of safe references

Safe references are not only used for argument passing. When you
destructure on a value in an `alt` expression, or loop over a vector
with `for`, variables bound to the inside of the given data structure
will use safe references, not copies. This means such references are
very cheap, but you'll occasionally have to copy them to ensure
safety.

    let my_rec = {a: 4, b: [1, 2, 3]};
    alt my_rec {
      {a, b} {
        log(info, b); // This is okay
        my_rec = {a: a + 1, b: b + [a]};
        log(info, b); // Here reference b has become invalid
      }
    }

## Argument passing styles

The fact that arguments are conceptually passed by safe reference does
not mean all arguments are passed by pointer. Composite types like
records and tags *are* passed by pointer, but single-word values, like
integers and pointers, are simply passed by value. Most of the time,
the programmer does not have to worry about this, as the compiler will
simply pick the most efficient passing style. There is one exception,
which will be described in the section on [generics](generic.html).

To explicitly set the passing-style for a parameter, you prefix the
argument name with a sigil. There are two special passing styles that
are often useful. The first is by-mutable-pointer, written with a
single `&`:

    fn vec_push(&v: [int], elt: int) {
        v += [elt];
    }

This allows the function to mutate the value of the argument, *in the
caller's context*. Clearly, you are only allowed to pass things that
can actually be mutated to such a function.

Then there is the by-copy style, written `+`. This indicates that the
function wants to take ownership of the argument value. If the caller
does not use the argument after the call, it will be 'given' to the
callee. Otherwise a copy will be made. This mode is mostly used for
functions that construct data structures. The argument will end up
being owned by the data structure, so if that can be done without a
copy, that's a win.

    type person = {name: str, address: str};
    fn make_person(+name: str, +address: str) -> person {
        ret {name: name, address: address};
    }
