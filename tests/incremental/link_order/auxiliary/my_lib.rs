//@ no-prefer-dynamic
//@[cfail1] compile-flags: -lbar -lfoo --crate-type lib -Zassert-incr-state=not-loaded
//@[cfail2] compile-flags: -lfoo -lbar --crate-type lib -Zassert-incr-state=not-loaded
