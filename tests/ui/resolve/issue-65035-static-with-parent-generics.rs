fn f<T>() {
    extern "C" {
        static a: *const T;
        //~^ ERROR can't use generic parameters from outer item
    }
}

fn g<T: Default>() {
    static a: *const T = Default::default();
    //~^ ERROR can't use generic parameters from outer item
}

fn h<const N: usize>() {
    extern "C" {
        static a: [u8; N];
        //~^ ERROR can't use generic parameters from outer item
    }
}

fn i<const N: usize>() {
    static a: [u8; N] = [0; N];
    //~^ ERROR can't use generic parameters from outer item
    //~| ERROR can't use generic parameters from outer item
}

fn main() {}
