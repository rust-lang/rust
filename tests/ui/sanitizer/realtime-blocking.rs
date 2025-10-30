//@ needs-sanitizer-support
//@ needs-sanitizer-realtime
//
//@ compile-flags: -Z sanitizer=realtime
//@ run-flags: abort-on-error=0
//
//@ run-fail
//@ error-pattern: Call to blocking function
//@ error-pattern: realtime_blocking::blocking::
//@ ignore-backends: gcc
#![feature(sanitize)]

#[sanitize(realtime = "nonblocking")]
fn sanitizer_on() {
    blocking();
}

#[sanitize(realtime = "blocking")]
fn blocking() {
    println!("blocking call not detected");
}

fn main() {
    sanitizer_on();
}
