unsafe fn foo<A>() {
    extern "C" {
        static baz: *const A;
        //~^ ERROR can't use generic parameters from outer item
    }

    let bar: *const u64 = core::mem::transmute(&baz);
}

fn main() { }
