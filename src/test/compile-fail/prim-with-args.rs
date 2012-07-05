fn main() {

let x: int<int>; //~ ERROR type parameters are not allowed on this type
let x: i8<int>; //~ ERROR type parameters are not allowed on this type
let x: i16<int>; //~ ERROR type parameters are not allowed on this type
let x: i32<int>; //~ ERROR type parameters are not allowed on this type
let x: i64<int>; //~ ERROR type parameters are not allowed on this type
let x: uint<int>; //~ ERROR type parameters are not allowed on this type
let x: u8<int>; //~ ERROR type parameters are not allowed on this type
let x: u16<int>; //~ ERROR type parameters are not allowed on this type
let x: u32<int>; //~ ERROR type parameters are not allowed on this type
let x: u64<int>; //~ ERROR type parameters are not allowed on this type
let x: float<int>; //~ ERROR type parameters are not allowed on this type
let x: char<int>; //~ ERROR type parameters are not allowed on this type

let x: int/&; //~ ERROR region parameters are not allowed on this type
let x: i8/&; //~ ERROR region parameters are not allowed on this type
let x: i16/&; //~ ERROR region parameters are not allowed on this type
let x: i32/&; //~ ERROR region parameters are not allowed on this type
let x: i64/&; //~ ERROR region parameters are not allowed on this type
let x: uint/&; //~ ERROR region parameters are not allowed on this type
let x: u8/&; //~ ERROR region parameters are not allowed on this type
let x: u16/&; //~ ERROR region parameters are not allowed on this type
let x: u32/&; //~ ERROR region parameters are not allowed on this type
let x: u64/&; //~ ERROR region parameters are not allowed on this type
let x: float/&; //~ ERROR region parameters are not allowed on this type
let x: char/&; //~ ERROR region parameters are not allowed on this type

}
