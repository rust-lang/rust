# `c_ffi_const`

The `#[c_ffi_const]` attribute applies clang's `const` attribute to foreign
functions declarations. 

That is, `#[c_ffi_const]` functions shall have no effects except for its return
value, which can only depend on the values of the function parameters, and is
not affected by changes to the observable state of the program.

The behavior of calling a `#[c_ffi_const]` function that violates these
requirements is undefined.

This attribute enables Rust to perform common optimizations, like sub-expression
elimination, and it can avoid emitting some calls in repeated invocations of the
function with the same argument values regardless of other operations being
performed in between these functions calls (as opposed to `#[c_ffi_pure]`
functions).

## Pitfalls

A `#[c_ffi_const]` function can only read global memory that would not affect
its return value for the whole execution of the program (e.g. immutable global
memory). `#[c_ffi_const]` functions are referentially-transparent and therefore
more strict than `#[c_ffi_pure]` functions.

A common pitfall involves applying the `#[c_ffi_const]` attribute to a
function that reads memory through pointer arguments which do not necessarily
point to immutable global memory.

A `#[c_ffi_const]` function that returns unit has no effect on the abstract
machine's state, and a `#[c_ffi_const]` function cannot be `#[c_ffi_pure]`.

A diverging and C or C++ `const` function is unreachable. Diverging via a
side-effect (e.g. a call to `abort`) violates `const` pre-conditions. Divergence
without side-effects is undefined behavior in C++ and not possible in C. In C++,
the behavior of infinite loops without side-effects is undefined, while in C
these loops can be assumed to terminate. This would cause a diverging function
to return, invoking undefined behavior.

When translating C headers to Rust FFI, it is worth verifying for which targets
the `const` attribute is enabled in those headers, and using the appropriate
`cfg` macros in the Rust side to match those definitions. While the semantics of
`const` are implemented identically by many C and C++ compilers, e.g., clang,
[GCC], [ARM C/C++ compiler], [IBM ILE C/C++], etc. they are not necessarily
implemented in this way on all of them. It is therefore also worth verifying
that the semantics of the C toolchain used to compile the binary being linked
against are compatible with those of the `#[c_ffi_const]`.

[ARM C/C++ compiler]: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491c/Cacgigch.html
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
[IBM ILE C/C++]: https://www.ibm.com/support/knowledgecenter/fr/ssw_ibm_i_71/rzarg/fn_attrib_const.htm
