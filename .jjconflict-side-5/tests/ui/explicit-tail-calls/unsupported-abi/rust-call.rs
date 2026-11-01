// extern "rust-call" untuples the first (and only, except self) argument, this requires additional
// support for tail calls to work correctly. Since there doesn't seem to be a usecase for tail
// calling extern "rust-call" functions (extern "rust-call" only shows up in closures, which we
// disallow tail-calling anyway), we just disallow that.
//
// @ ignore-backends: gcc
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]
#![feature(unboxed_closures)]

extern "rust-call" fn a(_: ()) {
    become a(());
    //~^ error: ABI does not support guaranteed tail calls
}

fn main() {}
