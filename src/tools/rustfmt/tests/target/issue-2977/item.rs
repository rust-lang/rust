macro_rules! atomic_bits {
    ($ldrex:expr) => {
        some_macro!(pub fn foo() {
            asm!($ldrex
                 : "=r"(raw)
                 : "r"(address)
                 :
                 : "volatile");
        })
    };
}
