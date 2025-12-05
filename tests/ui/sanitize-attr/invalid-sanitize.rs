//@ edition:2018
#![feature(sanitize)]

#[sanitize(brontosaurus = "off")] //~ ERROR malformed `sanitize` attribute input
fn main() {}

#[sanitize(address = "off")] //~ ERROR multiple `sanitize` attributes
#[sanitize(address = "off")]
fn multiple_consistent() {}

#[sanitize(address = "on")] //~ ERROR multiple `sanitize` attributes
#[sanitize(address = "off")]
fn multiple_inconsistent() {}

#[sanitize(address = "bogus")] //~ ERROR malformed `sanitize` attribute input
fn wrong_value() {}

#[sanitize = "off"] //~ ERROR malformed `sanitize` attribute input
fn name_value() {}

#[sanitize] //~ ERROR malformed `sanitize` attribute input
fn just_word() {}

#[sanitize(realtime = "on")] //~ ERROR malformed `sanitize` attribute input
fn wrong_value_realtime() {}

#[sanitize(realtime = "nonblocking")] //~ WARN: the async executor can run blocking code, without realtime sanitizer catching it [rtsan_nonblocking_async]
async fn async_nonblocking() {}

fn test() {
    let _async_block = {
        #[sanitize(realtime = "nonblocking")] //~ WARN: the async executor can run blocking code, without realtime sanitizer catching it [rtsan_nonblocking_async]
        async {}
    };

    let _async_closure = {
        #[sanitize(realtime = "nonblocking")] //~ WARN: the async executor can run blocking code, without realtime sanitizer catching it [rtsan_nonblocking_async]
        async || {}
    };

    let _regular_closure = {
        #[sanitize(realtime = "nonblocking")] // no warning on a regular closure
        || 0
    };
}
