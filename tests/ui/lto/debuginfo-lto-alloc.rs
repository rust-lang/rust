// With the upgrade to LLVM 16, the following error appeared when using
// link-time-optimization (LTO) alloc and debug compilation mode simultaneously:
//
//   error: Cannot represent a difference across sections
//
// The error stemmed from DI function definitions under type scopes, fixed by
// only declaring in type scope and defining the subprogram elsewhere.
// This test reproduces the circumstances that caused the error to appear, and checks
// that compilation is successful.

//@ build-pass
//@ compile-flags: --test -C debuginfo=2 -C lto=fat
//@ no-prefer-dynamic
//@ incremental
//@ ignore-backends: gcc

extern crate alloc;

#[cfg(test)]
mod tests {
    #[test]
    fn something_alloc() {
        assert_eq!(Vec::<u32>::new(), Vec::<u32>::new());
    }
}
