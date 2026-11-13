//@ no-prefer-dynamic
//@[bfail1] compile-flags: -lbar -lfoo --crate-type lib -Zassert-incr-state=not-loaded
//@[bfail2] compile-flags: -lfoo -lbar --crate-type lib -Zassert-incr-state=not-loaded
