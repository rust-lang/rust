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
