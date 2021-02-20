// compile-flags: --edition 2018

#![feature(try_blocks)]

pub fn main() {
    let res: Result<u32, std::array::TryFromSliceError> = try {
        Err("")?; //~ ERROR `?` couldn't convert the error to `TryFromSliceError`
        5
    };

    let res: Result<i32, i32> = try {
        "" //~ ERROR type mismatch
    };

    let res: Result<i32, i32> = try { }; //~ ERROR type mismatch

    let res: () = try { };
    //~^ ERROR the trait bound `(): Try2021` is not satisfied

    let res: i32 = try { 5 }; //~ ERROR the trait bound `i32: Try2021` is not satisfied
}
