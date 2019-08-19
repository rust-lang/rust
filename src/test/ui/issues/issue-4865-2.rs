// run-pass
// Previously, this would have failed to resolve due to the circular
// block between `use say` and `pub use hello::*`.
//
// Now, as `use say` is not `pub`, the glob import can resolve
// without any problem and this resolves fine.

pub use hello::*;

pub mod say {
    pub fn hello() { println!("hello"); }
}

pub mod hello {
    use say;

    pub fn hello() {
        say::hello();
    }
}

fn main() {
    hello();
}
