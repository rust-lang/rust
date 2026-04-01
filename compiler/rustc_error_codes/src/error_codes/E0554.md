Feature attributes are only allowed on the nightly release channel. Stable or
beta compilers will not comply.

Erroneous code example:

```ignore (depends on release channel)
#![feature(lang_items)] // error: `#![feature]` may not be used on the
                        //        stable release channel
```

If you need the feature, make sure to use a nightly release of the compiler
(but be warned that the feature may be removed or altered in the future).
