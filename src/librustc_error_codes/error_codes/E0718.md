A `#[lang = ".."]` attribute was placed on the wrong item type.

Erroneous code example:

```compile_fail,E0718
#![feature(lang_items)]

#[lang = "owned_box"]
static X: u32 = 42;
```
