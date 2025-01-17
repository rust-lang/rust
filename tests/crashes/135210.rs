//@ known-bug: #135210

#![feature(const_trait_impl)]
const _: fn(&String) = |s| {
    &*s as &str;
};

fn main() {}
