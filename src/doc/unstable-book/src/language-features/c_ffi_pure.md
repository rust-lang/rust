# `c_ffi_pure`

The `#[c_ffi_pure]` attribute applies clang's `pure` attribute to foreign
functions declarations. 

That is, `#[c_ffi_pure]` functions shall have no effects except for its return
value, which shall not change across two consecutive function calls and can only
depend on the values of the function parameters and/or global memory.

The behavior of calling a `#[c_ffi_pure]` function that violates these
requirements is undefined.

This attribute enables Rust to perform common optimizations, like sub-expression
elimination and loop optimizations. Some common examples of pure functions are
`strlen` or `memcmp`.

These optimizations only apply across successive invocations of the function,
since any other function could modify global memory read by `#[c_ffi_pure]`
functions, altering their result. The `#[c_ffi_const]` attribute allows
sub-expression elimination regardless of any operations in between the function
calls.

## Pitfals

A `#[c_ffi_pure]` function can read global memory through the function
parameters (e.g. pointers), globals, etc. `#[c_ffi_pure]` functions are not
referentially-transparent, and are therefore more relaxed than `#[c_ffi_const]`
functions.

However, accesing global memory through volatile or atomic reads can violate the
requirement that two consecutive function calls shall return the same value.

A `pure` function that returns unit has no effect on the abstract machine's
state.

A diverging and `pure` C or C++ function is unreachable. Diverging via a
side-effect (e.g. a call to `abort`) violates `pure` requirements. Divergence
without side-effects is undefined behavior in C++ and not possible in C. In C++,
the behavior of infinite loops without side-effects is undefined, while in C
these loops can be assumed to terminate. This would cause a diverging function
to return, invoking undefined behavior.

When translating C headers to Rust FFI, it is worth verifying for which targets
the `pure` attribute is enabled in those headers, and using the appropriate
`cfg` macros in the Rust side to match those definitions. While the semantics of
`pure` are implemented identically by many C and C++ compilers, e.g., clang,
[GCC], [ARM C/C++ compiler], [IBM ILE C/C++], etc. they are not necessarily
implemented in this way on all of them. It is therefore also worth verifying
that the semantics of the C toolchain used to compile the binary being linked
against are compatible with those of the `#[c_ffi_pure]`.


[ARM C/C++ compiler]: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491c/Cacigdac.html
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
[IBM ILE C/C++]: https://www.ibm.com/support/knowledgecenter/fr/ssw_ibm_i_71/rzarg/fn_attrib_pure.htm
