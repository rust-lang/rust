unsafe extern "Rust" {
    safe fn foo();
}

fn main() {
    foo(); //~ERROR: calling a `#[track_caller]` function from a non-`#[track_caller]` declaration
}

mod module {
    #[unsafe(no_mangle)]
    #[track_caller]
    fn foo() {
        panic!();
    }
}
