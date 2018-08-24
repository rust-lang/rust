#![feature(alloc)]
#![allow(unused_extern_crates)]

mod a {
    extern crate alloc;
    use alloc::HashMap;
    //~^ ERROR unresolved import `alloc` [E0432]
    //~| Did you mean `self::alloc`?
    mod b {
        use alloc::HashMap;
        //~^ ERROR unresolved import `alloc` [E0432]
        //~| Did you mean `a::alloc`?
        mod c {
            use alloc::HashMap;
            //~^ ERROR unresolved import `alloc` [E0432]
            //~| Did you mean `a::alloc`?
            mod d {
                use alloc::HashMap;
                //~^ ERROR unresolved import `alloc` [E0432]
                //~| Did you mean `a::alloc`?
            }
        }
    }
}
