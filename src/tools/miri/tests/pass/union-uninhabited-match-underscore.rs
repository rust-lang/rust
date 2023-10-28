fn main() {
    #[derive(Copy, Clone)]
    enum Void {}
    union Uninit<T: Copy> {
        value: T,
        uninit: (),
    }
    unsafe {
        let x: Uninit<Void> = Uninit { uninit: () };
        match x.value {
            // rustc warns about un unreachable pattern,
            // but is wrong in unsafe code.
            #[allow(unreachable_patterns)]
            _ => println!("hi from the void!"),
        }
    }
}
