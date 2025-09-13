//@ run-rustfix
mod foo {
    pub mod bar {
        pub mod baz {
            pub use std::vec::Vec as MyVec;
        }
    }
}

mod u {
    fn _a() {
        let _: Vec<i32> = super::foo::baf::baz::MyVec::new(); //~ ERROR cannot find
    }
}

mod v {
    fn _b() {
        let _: Vec<i32> = fox::bar::baz::MyVec::new(); //~ ERROR cannot find
    }
}

fn main() {
    let _t: Vec<i32> = vec::new(); //~ ERROR cannot find
    type _B = vec::Vec::<u8>; //~ ERROR cannot find
    let _t = std::sync_error::atomic::AtomicBool::new(true); //~ ERROR cannot find
}
