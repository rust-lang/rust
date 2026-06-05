//@ needs-asm-support

fn main() {
    std::arch::global_asm!("");
    //~^ ERROR using `global_asm!` in statement positions is experimental [E0658]
}
