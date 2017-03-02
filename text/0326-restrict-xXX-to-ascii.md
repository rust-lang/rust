- Start Date: 2014-09-26
- RFC PR: 326
- Rust Issue: https://github.com/rust-lang/rust/issues/18062

# Summary

In string literal contexts, restrict `\xXX` escape sequences to just
the range of ASCII characters, `\x00` -- `\x7F`.  `\xXX` inputs in
string literals with higher numbers are rejected (with an error
message suggesting that one use an `\uNNNN` escape).

# Motivation
[Motivation]: #motivation

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
 * https://github.com/rust-lang/meeting-minutes/blob/master/weekly-meetings/2014-01-21.md#xnn-escapes-in-strings
 * https://mail.mozilla.org/pipermail/rust-dev/2012-July/002025.html

Note in particular the meeting minutes bullet, where the team
explicitly decided to keep things "as they are".

However, at the time of that meeting, Rust did not have byte string
literals; people were converting string-literals into byte arrays via
the `bytes!` macro.  (Likewise, the rust-dev post is also from a time,
summer 2012, when we did not have byte-string literals.)

We are in a different world now.  The fact that now `\xXX` denotes a
code unit in a byte-string literal, but in a string literal denotes a
codepoint, does not seem elegant; it rather seems like a source of
confusion.  (Caveat: While Felix does believe this assertion, this
context-dependent interpretation of `\xXX` does have precedent
in both Python and Racket; see [Racket example] and [Python example]
appendices.)

By restricting `\xXX` to the range `0x00`--`0x7F`, we side-step the
question of "is it a code unit or a code point?" entirely (which was
the *real* context of both the rust-dev thread and the meeting minutes
bullet).  This RFC is a far more conservative choice that we can
safely make for the short term (i.e. for the 1.0 release) than it
would have been to switch to a "`\xXX` is a code unit" interpretation.

The expected outcome is reduced confusion for C/C++ programmers (which
is, after all, our primary target audience for conversion), and any
other language where `\xXX` never results in more than one byte.
The error message will point them to the syntax they need to adopt.

# Detailed design

In string literal contexts, `\xXX` inputs with `XX > 0x7F`  are
rejected (with an error message that mentions either, or both, of
`\uNNNN` escapes and the byte-string literal format `b".."`).

The full byte range remains supported when `\xXX` is used in
byte-string literals, `b"..."`

Raw strings by design do not offer escape sequences, so they are
unchanged.

Character and string escaping routines (such as
`core::char::escape_unicode`, and such as used by the `"{:?}"`
formatter) are updated so that string inputs that previously would
previously have printed `\xXX` with `XX > 0x7F` are updated to use
`\uNNNN` escapes instead.

# Drawbacks

Some reasons not to do this:

 * we think that the current behavior is intuitive,

 * it is consistent with language X (and thus has precedent),

 * existing libraries are relying on this behavior, or

 * we want to optimize for inputting characters with codepoints
   in the range above `0x7F` in string-literals, rather than
   optimizing for ASCII.

The thesis of this RFC is that the first bullet is a falsehood.

While there is some precedent for the "`\xXX` is code point"
interpretation in some languages, the [majority] do seem to favor the
"`\xXX` is code unit" point of view.  The proposal of this RFC is
side-stepping the distinction by limiting the input range for `\xXX`.

[majority]: https://mail.mozilla.org/pipermail/rust-dev/2012-July/002025.html

The third bullet is a strawman since we have not yet released 1.0, and
thus everything is up for change.

This RFC makes no comment on the validity of the fourth bullet.

# Alternatives

* We could remove `\xXX` entirely from string literals.  This would
  require people to use the `\uNNNN` escape format even for bytes in the
  range `00`--`0x7F`, which seems annoying.

* We could switch `\xXX` from meaning code point to meaning code unit
  in both string literal and byte-string literal contexts.  This
  was previously considered and explicitly rejected in an earlier
  meeting, as discussed in the [Motivation] section.

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

## Racket example
[Racket example]: #racket-example

```
% racket
Welcome to Racket v5.93.
> (define a-string "\xbb\n")
> (display a-string)
»
> (bytes-length (string->bytes/utf-8 a-string))
3
> (define a-byte-string #"\xc2\xbb\n")
> (bytes-length a-byte-string)
3
> (display a-byte-string)
»
> (exit)
%
```

The above code illustrates that in Racket, the `\xXX` escape sequence
denotes a code unit in byte-string context (`#".."` in that language),
while it denotes a code point in string context (`".."`).

## Python example
[Python example]: #python-example

```
% python
Python 2.7.5 (default, Mar  9 2014, 22:15:05)
[GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> a_string = u"\xbb\n";
>>> print a_string
»

>>> len(a_string.encode("utf-8"))
3
>>> a_byte_string = "\xc2\xbb\n";
>>> len(a_byte_string)
3
>>> print a_byte_string
»

>>> exit()
%
```

The above code illustrates that in Python, the `\xXX` escape sequence
denotes a code unit in byte-string context (`".."` in that language),
while it denotes a code point in *unicode* string context (`u".."`).
