# `patchable-function-entry`

--------------------

The `-Z patchable-function-entry=M,N` or `-Z patchable-function-entry=M`
compiler flag enables nop padding of function entries with M nops, with
an offset for the entry of the function at N nops. In the second form,
N defaults to 0.

As an illustrative example, `-Z patchable-function-entry=3,2` would produce:

```
nop
nop
function_label:
nop
//Actual function code begins here
```

This flag is used for hotpatching, especially in the Linux kernel. The flag
arguments are modeled after hte `-fpatchable-function-entry` flag as defined
for both [Clang](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fpatchable-function-entry)
and [gcc](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html#index-fpatchable-function-entry)
and is intended to provide the same effect.
