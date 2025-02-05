//@ compile-flags: -Z unstable-options --scrape-examples-output-path {{test-suite-build-base}}/t.calls --scrape-examples-target-crate foobar
//@ check-pass
#![no_std]
use core as _;
