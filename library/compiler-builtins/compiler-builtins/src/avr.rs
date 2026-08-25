intrinsics! {
    pub unsafe extern "C" fn abort() -> ! {
        // On AVRs, an architecture that doesn't support traps, unreachable code
        // paths get lowered into calls to `abort`:
        //
        // https://github.com/llvm/llvm-project/blob/cbe8f3ad7621e402b050e768f400ff0d19c3aedd/llvm/lib/CodeGen/SelectionDAG/LegalizeDAG.cpp#L4462
        //
        // When control gets here, it means that either core::intrinsics::abort()
        // was called or an undefined bebavior has occurred, so there's not that
        // much we can do to recover - we can't `panic!()`, because for all we
        // know the environment is gone now, so panicking might end up with us
        // getting back to this very function.
        //
        // So let's do the next best thing, loop.
        //
        // Alternatively we could (try to) restart the program, but since
        // undefined behavior is undefined, there's really no obligation for us
        // to do anything here - for all we care, we could just set the chip on
        // fire; but that'd be bad for the environment.

        loop {}
    }
}
