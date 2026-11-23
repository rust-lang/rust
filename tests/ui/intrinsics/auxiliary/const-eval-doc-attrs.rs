#![feature(type_info)]

use std::any::TypeId;

///CCC
#[doc = "DDD"]
pub struct C;

pub const N: usize = {
    struct TestStruct {
        /// aaa
        #[doc = "bbb"]
        field: u8,
    }
    let ty_id = TypeId::of::<TestStruct>();
    ty_id.field(0, 0).docs().len()
};

const _: () = assert!( N == 0 );
