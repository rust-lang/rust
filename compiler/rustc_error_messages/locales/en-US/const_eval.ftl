const-eval-unstable-in-stable =
    const-stable function cannot use `#[feature({$gate})]`
    .unstable-sugg = if it is not part of the public API, make this function unstably const
    .bypass-sugg = otherwise `#[rustc_allow_const_fn_unstable]` can be used to bypass stability checks
