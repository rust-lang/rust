//@ only-x86_64
//@ needs-asm-support
//@ run-pass

// This makes sure that even though `foo()` is instantiated multiple times, `bar()` is not.

#![feature(global_asm_statement_position)]

fn foo<T: std::fmt::Display>(value: T) {
    std::arch::global_asm!(".global bar", "bar:", "ret");

    println!("{value}");
}

unsafe extern "C" {
    safe fn bar();
}

fn main() {
    foo("test");
    foo(42);
    bar();
}
