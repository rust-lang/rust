#![allow(non_snake_case)]

use syntax::{register_diagnostic, register_long_diagnostics};

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {
E0660: r##"
The argument to the `asm` macro is not well-formed.

Erroneous code example:

```compile_fail,E0660
asm!("nop" "nop");
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0661: r##"
An invalid syntax was passed to the second argument of an `asm` macro line.

Erroneous code example:

```compile_fail,E0661
let a;
asm!("nop" : "r"(a));
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0662: r##"
An invalid input operand constraint was passed to the `asm` macro (third line).

Erroneous code example:

```compile_fail,E0662
asm!("xor %eax, %eax"
     :
     : "=test"("a")
    );
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0663: r##"
An invalid input operand constraint was passed to the `asm` macro (third line).

Erroneous code example:

```compile_fail,E0663
asm!("xor %eax, %eax"
     :
     : "+test"("a")
    );
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0664: r##"
A clobber was surrounded by braces in the `asm` macro.

Erroneous code example:

```compile_fail,E0664
asm!("mov $$0x200, %eax"
     :
     :
     : "{eax}"
    );
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0665: r##"
The `Default` trait was derived on an enum.

Erroneous code example:

```compile_fail,E0665
#[derive(Default)]
enum Food {
    Sweet,
    Salty,
}
```

The `Default` cannot be derived on an enum for the simple reason that the
compiler doesn't know which value to pick by default whereas it can for a
struct as long as all its fields implement the `Default` trait as well.

If you still want to implement `Default` on your enum, you'll have to do it "by
hand":

```
enum Food {
    Sweet,
    Salty,
}

impl Default for Food {
    fn default() -> Food {
        Food::Sweet
    }
}
```
"##,
}
