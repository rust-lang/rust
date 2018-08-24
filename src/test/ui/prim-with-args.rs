fn main() {

let x: isize<isize>; //~ ERROR type parameters are not allowed on this type
let x: i8<isize>; //~ ERROR type parameters are not allowed on this type
let x: i16<isize>; //~ ERROR type parameters are not allowed on this type
let x: i32<isize>; //~ ERROR type parameters are not allowed on this type
let x: i64<isize>; //~ ERROR type parameters are not allowed on this type
let x: usize<isize>; //~ ERROR type parameters are not allowed on this type
let x: u8<isize>; //~ ERROR type parameters are not allowed on this type
let x: u16<isize>; //~ ERROR type parameters are not allowed on this type
let x: u32<isize>; //~ ERROR type parameters are not allowed on this type
let x: u64<isize>; //~ ERROR type parameters are not allowed on this type
let x: char<isize>; //~ ERROR type parameters are not allowed on this type

let x: isize<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: i8<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: i16<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: i32<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: i64<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: usize<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: u8<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: u16<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: u32<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: u64<'static>; //~ ERROR lifetime parameters are not allowed on this type
let x: char<'static>; //~ ERROR lifetime parameters are not allowed on this type

}
