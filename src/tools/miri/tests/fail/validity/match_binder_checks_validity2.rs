fn main() {
    #[derive(Copy, Clone)]
    union Uninit<T: Copy> {
        value: T,
        uninit: u8,
    }
    unsafe {
        let x: Uninit<bool> = Uninit { uninit: 3 };
        match x.value {
            #[allow(unreachable_patterns)]
            _x => println!("hi from the void!"), //~ERROR: invalid value
        }
    }
}
