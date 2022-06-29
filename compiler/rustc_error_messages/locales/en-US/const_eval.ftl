const-eval-unstable-in-stable =
    const-stable function cannot use `#[feature({$gate})]`
    .unstable-sugg = if it is not part of the public API, make this function unstably const
    .bypass-sugg = otherwise `#[rustc_allow_const_fn_unstable]` can be used to bypass stability checks

const-eval-thread-local-access =
    thread-local statics cannot be accessed at compile-time

const-eval-static-access =
    { $kind -> 
        [constant function] constant functions
        [static] statics
        *[constant] constants
    } cannot refer to statics
    .help = consider extracting the value of the `static` to a `const`, and referring to that
    .teach-note = `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
    .teach-help = To fix this, the value can be extracted to a `const` and then used.

const-eval-raw-ptr-to-int =
    pointers cannot be cast to integers during const eval
    .note = at compile-time, pointers do not have an integer value
    .note2 = avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior

const-eval-raw-ptr-comparison =
    pointers cannot be reliably compared during const eval
    .note = see issue #53020 <https://github.com/rust-lang/rust/issues/53020> for more information

const-eval-panic-non-str = argument to `panic!()` in a const context must have type `&str`

const-eval-mut-deref =
    mutation through a reference is not allowed in { $kind -> 
        [constant function] constant functions
        [static] statics
        *[constant] constants
    }

const-eval-transient-mut-borrow = mutable references are not allowed in { $kind -> 
        [constant function] constant functions
        [static] statics
        *[constant] constants
    }

const-eval-transient-mut-borrow-raw = raw mutable references are not allowed in { $kind -> 
        [constant function] constant functions
        [static] statics
        *[constant] constants
    }
