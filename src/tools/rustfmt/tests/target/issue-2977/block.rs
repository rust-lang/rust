macro_rules! atomic_bits {
    ($ldrex:expr) => {
        execute(|| {
            asm!($ldrex
                 : "=r"(raw)
                 : "r"(address)
                 :
                 : "volatile");
        })
    };
}
