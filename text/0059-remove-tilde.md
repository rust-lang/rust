- Start Date: 2014-04-30
- RFC PR: [rust-lang/rfcs#59](https://github.com/rust-lang/rfcs/pull/59)
- Rust Issue: [rust-lang/rust#13885](https://github.com/rust-lang/rust/issues/13885)

# Summary

The tilde (`~`) operator and type construction do not support allocators and therefore should be removed in favor of the `box` keyword and a language item for the type.

# Motivation

* There will be a unique pointer type in the standard library, `Box<T,A>` where `A` is an allocator. The `~T` type syntax does not allow for custom allocators. Therefore, in order to keep `~T` around while still supporting allocators, we would need to make it an alias for `Box<T,Heap>`. In the spirit of having one way to do things, it seems better to remove `~` entirely as a type notation.

* `~EXPR` and `box EXPR` are duplicate functionality; the former does not support allocators. Again in the spirit of having one and only one way to do things, I would like to remove `~EXPR`.

* Some people think `~` is confusing, as it is less self-documenting than `Box`.

* `~` can encourage people to blindly add sigils attempting to get their code to compile instead of consulting the library documentation.

# Drawbacks

`~T` may be seen as convenient sugar for a common pattern in some situations.

# Detailed design

The `~EXPR` production is removed from the language, and all such uses are converted into `box`.

Add a lang item, `box`. That lang item will be defined in `liballoc` (NB: not `libmetal`/`libmini`, for bare-metal programming) as follows:

    #[lang="box"]
    pub struct Box<T,A=Heap>(*T);

All parts of the compiler treat instances of `Box<T>` identically to the way it treats `~T` today.

The destructuring form for `Box<T>` will be `box PAT`, as follows:

    let box(x) = box(10);
    println!("{}", x); // prints 10

# Alternatives

The other possible design here is to keep `~T` as sugar. The impact of doing this would be that a common pattern would be terser, but I would like to not do this for the reasons stated in "Motivation" above.

# Unresolved questions

The allocator design is not yet fully worked out.

It may be possible that unforeseen interactions will appear between the struct nature of `Box<T>` and the built-in nature of `~T` when merged.
