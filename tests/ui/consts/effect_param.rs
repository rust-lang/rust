//! Ensure we don't allow accessing const effect parameters from stable Rust.

fn main() {
    i8::checked_sub::<true>(42, 43);
    //~^ ERROR: method takes 0 generic arguments but 1 generic argument was supplied
    i8::checked_sub::<false>(42, 43);
    //~^ ERROR: method takes 0 generic arguments but 1 generic argument was supplied
}

const FOO: () = {
    i8::checked_sub::<false>(42, 43);
    //~^ ERROR: method takes 0 generic arguments but 1 generic argument was supplied
    i8::checked_sub::<true>(42, 43);
    //~^ ERROR: method takes 0 generic arguments but 1 generic argument was supplied
};
