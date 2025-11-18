# Debugger Visualizers

These are typically the last step before the debugger displays the information, but the results may
be piped through a debug adapter such as an IDE's debugger API.

The term "Visualizer" is a bit of a misnomer. The real goal isn't just to prettify the output, but
to provide an interface for the user to interact with that is as useful as possible. In many cases
this means reconstructing the original type as closely as possible to its Rust representation, but
not always.

The visualizer interface allows generating "synthetic children" - fields that don't exist in the
debug info, but can be derived from invariants about the language and the type itself. A simple
example is allowing one to interact with the elements of a `Vec<T>` instead of just it's `*mut u8`
heap pointer, length, and capacity.

## Performance

Before tackling the visualizers themselves, it's important to note that these are part of a
performance-sensitive system. Please excuse the break in formality, but: if I have to spend
significant time debugging, I'm annoyed. If I have to *wait on my debugger*, I'm pissed.

Every millisecond spent in these visualizers is a millisecond longer for the user to see output.
This can be especially painful for large stackframes that contain many/large container types.
Debugger GUI's such as VSCode will request the whole stack frame at once, and this can result in
delays of tens of seconds (or even minutes) before being able to interact with any variables in the
frame.

There is a tendancy to balk at the idea of optimizing Python code, but it really can have a
substantial impact. Remember, there is no compiler to help keep the code fast. Even simple
transformations are not done for you. It can be difficult to find Python performance tips through
all the noise of people suggesting you don't bother optimizing Python, so here are some things to
keep in mind that are relevant to these scripts:

* Everything allocates, even `int`
* Use tuples when possible. `list` is effectively `Vec<Box<[Any]>>`, whereas tuples are equivalent
to `Box<[Any]>`. They have one less layer of indirection, don't carry extra capacity and can't
grow/shrink which can be advantageous in many cases. An additional benefit is that Python caches and
recycles the underlying allocations of all tuples up to size 20.
* Regexes are slow and should be avoided when simple string manipulation will do
* Strings are immutable, thus many string operations implictly copy the contents.
* When concatenating large lists of strings, `"".join(iterable_of_strings)` is typically the fastest
way to do it.
* f-strings are generally the fastest way to do small, simple string transformations such as
surrounding a string with parentheses.
* The act of calling a function is somewhat slow (even if the function is completely empty). If the
code section is very hot, consider inlining the function manually.
* Local variable access is significantly faster than global and built-in function access
* Member/method access via the `.` operator is also slow, consider reassigning deeply nested values
to local variables to avoid this cost (e.g. `h = a.b.c.d.e.f.g.h`).
* Accessing inherited methods and fields is about 2x slower than base-class methods and fields.
Avoid inheritance whenever possible.
* Use [`__slots__`](https://wiki.python.org/moin/UsingSlots) wherever possible. `__slots__` is a way
to indicate to Python that your class's fields won't change and speeds up field access by a
noticable amount. This does require you to name your fields in advance and initialize them in
`__init__`, but it's a small price to pay for the benefits.
* Match statements/if..elif..else are not optimized in any way. The conditions are checked in order,
1 by 1. If possible, use an alternative such as dictionary dispatch or a table of values
* Compute lazily when possible
* List comprehensions are typically faster than loops, generator comprehensions are a bit slower
than list comprehensions, but use less memory. You can think of comprehensions as equivalent to
Rust's `iter.map()`. List comprehensions effectively call `collect::<Vec<_>>` at the end, whereas
generator comprehensions do not.