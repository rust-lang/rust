unsafe extern "Rust" {
    #[track_caller]
    safe fn foo();
}

fn main() {
    foo(); //~ERROR: calling a non-`#[track_caller]` function from a `#[track_caller]` declaration
}

mod module {
    #[unsafe(no_mangle)]
    fn foo() {
        panic!();
    }
}
