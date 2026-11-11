macro_rules! atomic_bits {
    // the println macro cannot be rewritten because of the asm macro
    ($type:ty, $ldrex:expr, $strex:expr) => {
        impl AtomicBits for $type {
            unsafe fn load_excl(address: usize) -> Self {
                let raw: $type;
                asm!($ldrex
                     : "=r"(raw)
                     : "r"(address)
                     :
                     : "volatile");
                raw
            }

            unsafe fn store_excl(self, address: usize) -> bool {
                let status: $type;
                println!("{}",
                         status);
                status == 0
            }
        }
    };

    // the println macro should be rewritten here
    ($type:ty) => {
        fn some_func(self) {
            let status: $type;
            println!("{}", status);
        }
    };

    // unrewritale macro in func
    ($type:ty, $ldrex:expr) => {
        unsafe fn load_excl(address: usize) -> Self {
            let raw: $type;
            asm!($ldrex
                 : "=r"(raw)
                 : "r"(address)
                 :
                 : "volatile");
            raw
        }
    }
}
