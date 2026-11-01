//@ run-pass
//@ only-x86_64-unknown-linux-gnu
//@ ignore-backends: gcc

// https://github.com/rust-lang/rust/issues/151950

unsafe extern "C" {
    #[link_name = "exit@GLIBC_2.2.5"]
    safe fn exit(status: i32) -> !;
    safe fn my_exit(status: i32) -> !;
}

core::arch::global_asm!(".global my_exit", "my_exit:", "jmp {}", sym exit);

fn main() {
    my_exit(0);
}
