An unsupported naked function definition.

Erroneous code example:

```compile_fail,E0787
#![feature(naked_functions)]

#[naked]
pub extern "C" fn f() -> u32 {
    42
}
```

The naked functions must be defined using a single inline assembly
block.

The execution must never fall through past the end of the assembly
code so the block must use `noreturn` option. The asm block can also
use `att_syntax` and `raw` options, but others options are not allowed.

The asm block must not contain any operands other than `const` and
`sym`.

### Additional information

For more information, please see [RFC 2972].

[RFC 2972]: https://github.com/rust-lang/rfcs/blob/master/text/2972-constrained-naked.md
