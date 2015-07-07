% Constructors

Unlike C++, Rust does not come with a slew of builtin
kinds of constructor. There are no Copy, Default, Assignment, Move, or whatever constructors.
This largely has to do with Rust's philosophy of being explicit.

Move constructors are meaningless in Rust because we don't enable types to "care" about their
location in memory. Every type must be ready for it to be blindly memcopied to somewhere else
in memory. This means pure on-the-stack-but-still-movable intrusive linked lists are simply
not happening in Rust (safely).

Assignment and copy constructors similarly don't exist because move semantics are the *default*
in rust. At most `x = y` just moves the bits of y into the x variable. Rust does provide two
facilities for going back to C++'s copy-oriented semantics: `Copy` and `Clone`. Clone is our
moral equivalent of a copy constructor, but it's never implicitly invoked. You have to explicitly
call `clone` on an element you want to be cloned. Copy is a special case of Clone where the
implementation is just "copy the bits". Copy types *are* implicitly
cloned whenever they're moved, but because of the definition of Copy this just means *not*
treating the old copy as uninitialized -- a no-op.

While Rust provides a `Default` trait for specifying the moral equivalent of a default
constructor, it's incredibly rare for this trait to be used. This is because variables
[aren't implicitly initialized][uninit]. Default is basically only useful for generic
programming. In concrete contexts, a type will provide a static `new` method for any
kind of "default" constructor. This has no relation to `new` in other
languages and has no special meaning. It's just a naming convention.