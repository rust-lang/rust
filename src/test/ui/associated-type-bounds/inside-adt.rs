// compile-fail
#![feature(associated_type_bounds)]
#![feature(untagged_unions)]

struct S1 { f: dyn Iterator<Item: Copy> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses
struct S2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses
struct S3 { f: dyn Iterator<Item: 'static> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses

enum E1 { V(dyn Iterator<Item: Copy>) }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses
enum E2 { V(Box<dyn Iterator<Item: Copy>>) }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses
enum E3 { V(dyn Iterator<Item: 'static>) }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses

union U1 { f: dyn Iterator<Item: Copy> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses
union U2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses
union U3 { f: dyn Iterator<Item: 'static> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR could not find defining uses

fn main() {}
