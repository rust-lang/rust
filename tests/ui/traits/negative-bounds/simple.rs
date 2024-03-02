#![feature(negative_bounds, negative_impls)]

fn not_copy<T: !Copy>() {}

fn neg_param_env<T: !Copy>() {
    not_copy::<T>();
}

fn pos_param_env<T: Copy>() {
    not_copy::<T>();
    //~^ ERROR : !Copy` is not satisfied
}

fn unknown<T>() {
    not_copy::<T>();
    //~^ ERROR : !Copy` is not satisfied
}

struct NotCopyable;
impl !Copy for NotCopyable {}

fn neg_impl() {
    not_copy::<NotCopyable>();
}

#[derive(Copy, Clone)]
struct Copyable;

fn pos_impl() {
    not_copy::<Copyable>();
    //~^ ERROR : !Copy` is not satisfied
}

struct NotNecessarilyCopyable;

fn unknown_impl() {
    not_copy::<NotNecessarilyCopyable>();
    //~^ ERROR : !Copy` is not satisfied
}

fn main() {}
