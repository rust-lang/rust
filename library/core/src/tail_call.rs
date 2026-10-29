/// How to proceed with evaluating a chain to tail calls.
#[lang = "tail_next"]
#[doc(hidden)]
#[unstable(feature = "tail_call_fallback", issue = "none")]
#[derive(Debug)]
pub enum TailNext<Args, Ret> {
    /// The call chain has bottomed out.
    #[lang = "tail_next_done"]
    Done(Ret),
    /// The call chain continues.
    #[lang = "tail_next_call"]
    Call(fn(Args) -> TailNext<Args, Ret>, Args),
}

/// Repeatedly call the (shim) function until done.
#[lang = "tail_eval"]
#[doc(hidden)]
#[unstable(feature = "tail_call_fallback", issue = "none")]
pub fn tail_eval<Args, Ret>(mut f: fn(Args) -> TailNext<Args, Ret>, mut args: Args) -> Ret {
    loop {
        match f(args) {
            TailNext::Call(next_f, next_args) => {
                f = next_f;
                args = next_args;
            }
            TailNext::Done(ret) => return ret,
        }
    }
}

#[lang = "tail_shim"]
#[doc(hidden)]
#[unstable(feature = "tail_call_fallback", issue = "none")]
pub fn tail_shim<F, Args, Ret>(_args: Args) -> TailNext<Args, Ret> {
    // Replaced by the real shim by the compiler; never actually called.
    loop {}
}
