//@ edition: 2021

#![feature(async_closure)]

pub fn test(test: &()) {
    async |unconstrained| {
        //~^ ERROR type annotations needed
        (test,)
    };
}

fn main() {}
