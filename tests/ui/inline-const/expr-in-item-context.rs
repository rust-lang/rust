// reported in https://github.com/rust-lang/rust/issues/128338

// This had an overly-terse diagnostic, because we recognized this is where an item should be, but
// but didn't recognize an inline const in this place is probably trying to be an "anon const item".
// Those are written like this:
const _: () = {};
// const _ are often used as compile-time assertions that don't conflict with other const items

const { assert!(size_of::<u32>() <= size_of::<usize>()) };
//~^ expected item, found keyword
//~| to evaluate a const expression, use an anonymous const

fn main() {
    const { assert!(size_of::<u32>() <= size_of::<usize>()) };
}
