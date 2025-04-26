//@ known-bug: #122259

#![feature(unsized_fn_params)]

#[derive(Copy, Clone)]
struct Target(str);

fn w(t: &Target) {
    x(*t);
}

fn x(t: Target) {}
