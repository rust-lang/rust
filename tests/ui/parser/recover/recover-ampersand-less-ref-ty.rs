//@ edition: 2021

struct Entity<'a> {
    name: 'a str, //~ ERROR expected type, found lifetime
    //~^ HELP you might have meant to write a reference type here
}

struct Buffer<'buf> {
    bytes: 'buf mut [u8], //~ ERROR expected type, found lifetime
    //~^ HELP you might have meant to write a reference type here
}

fn main() {}
