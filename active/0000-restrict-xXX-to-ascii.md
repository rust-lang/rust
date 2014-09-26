- Start Date: 2014-09-26
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

In string literal contexts, restrict `\xXX` escape sequences to just
the range of ASCII characters, `\x00` -- `\x7F`.  `\xXX` inputs with
higher numbers are rejected (with an error message suggesting that one
use an `\uNNNN` escape).

# Motivation

In a string literal context, the current `\xXX` character escape
sequence is potentially confusing when given inputs greater than
`0x7F`, because it does not encode that byte literally, but instead
encodes whatever the escape sequence `\u00XX` would produce.

Thus, for inputs greater than `0x7F`, `\xXX` will encode multiple
bytes into the generated string literal, as illustrated in the
[Rust example] appendix.

This is different from what C/C++ programmers might expect (see
[Behavior of xXX in C] appendix).

(It would not be legal to encode the single byte literally into the
string literal, since then the string would not be well-formed UTF-8.)

It has been suggested that the `\xXX` character escape should be
removed entirely (at least from string literal contexts).  This RFC is
taking a slightly less aggressive stance: keep `\xXX`, but only for
ASCII inputs when it occurs in string literals.  This way, people can
continue using this escape format (which shorter than the `\uNNNN`
format) when it makes sense.

Here are some links to discussions on this topic, including direct
comments that suggest exactly the strategy of this RFC.

 * https://github.com/rust-lang/rfcs/issues/312
 * https://github.com/rust-lang/rust/issues/12769
 * https://github.com/rust-lang/rust/issues/2800#issuecomment-31477259
 * https://github.com/rust-lang/rfcs/pull/69#issuecomment-43002505
 * https://github.com/rust-lang/rust/issues/12769#issuecomment-43574856

The expected outcome is reduced confusion for C/C++ programmers, and
any other language where `\xXX` never results in more than one byte.

# Detailed design

In string literal contexts, `\xXX` inputs with higher numbers are
rejected (with an error message that mentions either, or both, of
`\uNNNN` escapes and the byte-string literal format `b".."`).

The full byte range remains supported when `\xXX` is used in
byte-string literals, `b"..."`

Raw strings by design do not offer escape sequences, so they are
unchanged.

Character and string escaping routines (such as
`core::char::escape_unicode`, and such as used by the "{:?}"
formatter) are updated so that string inputs that previously would
previously have printed `\xXX` with `XX > 0x7F` are updated to use
`\uNNNN` escapes instead.

# Drawbacks

The only reasons not to do this are either:

 * we think that the current behavior is intuitive, or

 * existing libraries are relying on this behavior.

The thesis of this RFC is that the first bullet is a falsehood.

The latter bullet is a strawman since we have not yet released 1.0,
and thus everything is up for change.

# Alternatives

We could remove `\xXX` entirely from string literals.  This would
require people to use the `\uNNNN` escape format even for bytes in the
range `00`--`0x7F`, which seems annoying.

# Unresolved questions

None.

# Appendices

## Behavior of xXX in C
[Behavior of xXX in C]: #behavior-of-xxx-in-c

Here is a C program illustrating how `xXX` escape sequences are treated
in string literals in that context:

```c
#include <stdio.h>

int main() {
    char *s;

    s = "a";
    printf("s[0]: %d\n", s[0]);
    printf("s[1]: %d\n", s[1]);

    s = "\x61";
    printf("s[0]: %d\n", s[0]);
    printf("s[1]: %d\n", s[1]);

    s = "\x7F";
    printf("s[0]: %d\n", s[0]);
    printf("s[1]: %d\n", s[1]);

    s = "\x80";
    printf("s[0]: %d\n", s[0]);
    printf("s[1]: %d\n", s[1]);
    return 0;
}
```

Its output is the following:
```
% gcc example.c && ./a.out
s[0]: 97
s[1]: 0
s[0]: 97
s[1]: 0
s[0]: 127
s[1]: 0
s[0]: -128
s[1]: 0
```

## Rust example
[Rust example]: #rust-example

Here is a Rust program that explores the various ways `\xXX` sequences are
treated in both string literal and byte-string literal contexts.

```rust
 #![feature(macro_rules)]

fn main() {
    macro_rules! print_str {
        ($r:expr, $e:expr) => { {
            println!("{:>20}: \"{}\"",
                     format!("\"{}\"", $r),
                     $e.escape_default())
        } }
    }

    macro_rules! print_bstr {
        ($r:expr, $e:expr) => { {
            println!("{:>20}: {}",
                     format!("b\"{}\"", $r),
                     $e)
        } }
    }

    macro_rules! print_bytes {
        ($r:expr, $e:expr) => {
            println!("{:>9}.as_bytes(): {}", format!("\"{}\"", $r), $e.as_bytes())
        } }

    // println!("{}", b"\u0000"); // invalid: \uNNNN is not a byte escape.
    print_str!(r"\0", "\0");
    print_bstr!(r"\0", b"\0");
    print_bstr!(r"\x00", b"\x00");
    print_bytes!(r"\x00", "\x00");
    print_bytes!(r"\u0000", "\u0000");
    println!("");
    print_str!(r"\x61", "\x61");
    print_bstr!(r"a", b"a");
    print_bstr!(r"\x61", b"\x61");
    print_bytes!(r"\x61", "\x61");
    print_bytes!(r"\u0061", "\u0061");
    println!("");
    print_str!(r"\x7F", "\x7F");
    print_bstr!(r"\x7F", b"\x7F");
    print_bytes!(r"\x7F", "\x7F");
    print_bytes!(r"\u007F", "\u007F");
    println!("");
    print_str!(r"\x80", "\x80");
    print_bstr!(r"\x80", b"\x80");
    print_bytes!(r"\x80", "\x80");
    print_bytes!(r"\u0080", "\u0080");
    println!("");
    print_str!(r"\xFF", "\xFF");
    print_bstr!(r"\xFF", b"\xFF");
    print_bytes!(r"\xFF", "\xFF");
    print_bytes!(r"\u00FF", "\u00FF");
    println!("");
    print_str!(r"\u0100", "\u0100");
    print_bstr!(r"\x01\x00", b"\x01\x00");
    print_bytes!(r"\u0100", "\u0100");
}
```

In current Rust, it generates output as follows:
```
% rustc --version && echo && rustc example.rs && ./example
rustc 0.12.0-pre (d52d0c836 2014-09-07 03:36:27 +0000)

                "\0": "\x00"
               b"\0": [0]
             b"\x00": [0]
   "\x00".as_bytes(): [0]
 "\u0000".as_bytes(): [0]

              "\x61": "a"
                b"a": [97]
             b"\x61": [97]
   "\x61".as_bytes(): [97]
 "\u0061".as_bytes(): [97]

              "\x7F": "\x7f"
             b"\x7F": [127]
   "\x7F".as_bytes(): [127]
 "\u007F".as_bytes(): [127]

              "\x80": "\x80"
             b"\x80": [128]
   "\x80".as_bytes(): [194, 128]
 "\u0080".as_bytes(): [194, 128]

              "\xFF": "\xff"
             b"\xFF": [255]
   "\xFF".as_bytes(): [195, 191]
 "\u00FF".as_bytes(): [195, 191]

            "\u0100": "\u0100"
         b"\x01\x00": [1, 0]
 "\u0100".as_bytes(): [196, 128]
%
```

Note that the behavior of `\xXX` on byte-string literals matches the
expectations established by the C program in [Behavior of xXX in C];
that is good.  The problem is the behavior of `\xXX` for `XX > 0x7F`
in string-literal contexts, namely in the fourth and fifth examples
where the `.as_bytes()` invocations are showing that the underlying
byte array has two elements instead of one.
