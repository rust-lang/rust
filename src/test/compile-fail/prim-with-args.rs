fn main() {

let x: int<int>; //~ ERROR Type parameters are not allowed on this type.
let x: i8<int>; //~ ERROR Type parameters are not allowed on this type.
let x: i16<int>; //~ ERROR Type parameters are not allowed on this type.
let x: i32<int>; //~ ERROR Type parameters are not allowed on this type.
let x: i64<int>; //~ ERROR Type parameters are not allowed on this type.
let x: uint<int>; //~ ERROR Type parameters are not allowed on this type.
let x: u8<int>; //~ ERROR Type parameters are not allowed on this type.
let x: u16<int>; //~ ERROR Type parameters are not allowed on this type.
let x: u32<int>; //~ ERROR Type parameters are not allowed on this type.
let x: u64<int>; //~ ERROR Type parameters are not allowed on this type.
let x: float<int>; //~ ERROR Type parameters are not allowed on this type.
let x: char<int>; //~ ERROR Type parameters are not allowed on this type.

let x: int/&; //~ ERROR Region parameters are not allowed on this type.
let x: i8/&; //~ ERROR Region parameters are not allowed on this type.
let x: i16/&; //~ ERROR Region parameters are not allowed on this type.
let x: i32/&; //~ ERROR Region parameters are not allowed on this type.
let x: i64/&; //~ ERROR Region parameters are not allowed on this type.
let x: uint/&; //~ ERROR Region parameters are not allowed on this type.
let x: u8/&; //~ ERROR Region parameters are not allowed on this type.
let x: u16/&; //~ ERROR Region parameters are not allowed on this type.
let x: u32/&; //~ ERROR Region parameters are not allowed on this type.
let x: u64/&; //~ ERROR Region parameters are not allowed on this type.
let x: float/&; //~ ERROR Region parameters are not allowed on this type.
let x: char/&; //~ ERROR Region parameters are not allowed on this type.

}
