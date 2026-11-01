//@ run-rustfix
fn main() {

let _x: isize<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: i8<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: i16<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: i32<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: i64<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: usize<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: u8<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: u16<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: u32<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: u64<isize>; //~ ERROR type arguments are not allowed on builtin type
let _x: char<isize>; //~ ERROR type arguments are not allowed on builtin type

let _x: isize<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: i8<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: i16<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: i32<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: i64<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: usize<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: u8<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: u16<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: u32<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: u64<'static>; //~ ERROR lifetime arguments are not allowed on builtin type
let _x: char<'static>; //~ ERROR lifetime arguments are not allowed on builtin type

}
