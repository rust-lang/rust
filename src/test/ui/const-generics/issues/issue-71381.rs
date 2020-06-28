#![feature(const_generics)]
#![allow(incomplete_features)]

struct Test(*const usize);

type PassArg = ();

unsafe extern "C" fn pass(args: PassArg) {
    println!("Hello, world!");
}

impl Test {
    pub fn call_me<Args: Sized, const IDX: usize, const FN: unsafe extern "C" fn(Args)>(&self) {
        //~^ ERROR: using function pointers as const generic parameters is forbidden
        self.0 = Self::trampiline::<Args, IDX, FN> as _
    }

    unsafe extern "C" fn trampiline<
        Args: Sized,
        const IDX: usize,
        const FN: unsafe extern "C" fn(Args),
        //~^ ERROR: using function pointers as const generic parameters is forbidden
    >(
        args: Args,
    ) {
        FN(args)
    }
}

fn main() {
    let x = Test();
    x.call_me::<PassArg, 30, pass>()
}
