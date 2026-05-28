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
            #[allow(unreachable_patterns)]
            _x => println!("hi from the void!"), //~ERROR: invalid value
        }
    }
}
