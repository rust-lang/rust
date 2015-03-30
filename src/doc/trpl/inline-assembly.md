% Inline Assembly

For extremely low-level manipulations and performance reasons, one
might wish to control the CPU directly. Rust supports using inline
assembly to do this via the `asm!` macro. The syntax roughly matches
that of GCC & Clang:

```ignore
asm!(assembly template
   : output operands
   : input operands
   : clobbers
   : options
   );
```

Any use of `asm` is feature gated (requires `#![feature(asm)]` on the
crate to allow) and of course requires an `unsafe` block.

> **Note**: the examples here are given in x86/x86-64 assembly, but
> all platforms are supported.

## Assembly template

The `assembly template` is the only required parameter and must be a
literal string (i.e. `""`)

```
#![feature(asm)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn foo() {
    unsafe {
        asm!("NOP");
    }
}

// other platforms
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn foo() { /* ... */ }

fn main() {
    // ...
    foo();
    // ...
}
```

(The `feature(asm)` and `#[cfg]`s are omitted from now on.)

Output operands, input operands, clobbers and options are all optional
but you must add the right number of `:` if you skip them:

```
# #![feature(asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() { unsafe {
asm!("xor %eax, %eax"
    :
    :
    : "eax"
   );
# } }
```

Whitespace also doesn't matter:

```
# #![feature(asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() { unsafe {
asm!("xor %eax, %eax" ::: "eax");
# } }
```

## Operands

Input and output operands follow the same format: `:
"constraints1"(expr1), "constraints2"(expr2), ..."`. Output operand
expressions must be mutable lvalues:

```
# #![feature(asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn add(a: i32, b: i32) -> i32 {
    let mut c = 0;
    unsafe {
        asm!("add $2, $0"
             : "=r"(c)
             : "0"(a), "r"(b)
             );
    }
    c
}
# #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
# fn add(a: i32, b: i32) -> i32 { a + b }

fn main() {
    assert_eq!(add(3, 14159), 14162)
}
```

## Clobbers

Some instructions modify registers which might otherwise have held
different values so we use the clobbers list to indicate to the
compiler not to assume any values loaded into those registers will
stay valid.

```
# #![feature(asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() { unsafe {
// Put the value 0x200 in eax
asm!("mov $$0x200, %eax" : /* no outputs */ : /* no inputs */ : "eax");
# } }
```

Input and output registers need not be listed since that information
is already communicated by the given constraints. Otherwise, any other
registers used either implicitly or explicitly should be listed.

If the assembly changes the condition code register `cc` should be
specified as one of the clobbers. Similarly, if the assembly modifies
memory, `memory` should also be specified.

## Options

The last section, `options` is specific to Rust. The format is comma
separated literal strings (i.e. `:"foo", "bar", "baz"`). It's used to
specify some extra info about the inline assembly:

Current valid options are:

1. *volatile* - specifying this is analogous to
   `__asm__ __volatile__ (...)` in gcc/clang.
2. *alignstack* - certain instructions expect the stack to be
   aligned a certain way (i.e. SSE) and specifying this indicates to
   the compiler to insert its usual stack alignment code
3. *intel* - use intel syntax instead of the default AT&T.

