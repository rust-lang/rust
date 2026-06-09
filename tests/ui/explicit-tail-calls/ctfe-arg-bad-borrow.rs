#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

pub const fn test(_: &Type) {
    const fn takes_borrow(_: &Type) {}

    let local = Type;
    become takes_borrow(&local);
    //~^ error: `local` does not live long enough
}

struct Type;

fn main() {}
