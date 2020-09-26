#![feature(associated_type_bounds)]
#![feature(untagged_unions)]

struct S1 { f: dyn Iterator<Item: Copy> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
struct S2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
struct S3 { f: dyn Iterator<Item: 'static> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions

enum E1 { V(dyn Iterator<Item: Copy>) }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR the size for values of type `(dyn Iterator<Item = impl Copy> + 'static)`
enum E2 { V(Box<dyn Iterator<Item: Copy>>) }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
enum E3 { V(dyn Iterator<Item: 'static>) }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR the size for values of type `(dyn Iterator<Item = impl Sized> + 'static)`

union U1 { f: dyn Iterator<Item: Copy> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR the size for values of type `(dyn Iterator<Item = impl Copy> + 'static)`
union U2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
union U3 { f: dyn Iterator<Item: 'static> }
//~^ ERROR associated type bounds are not allowed within structs, enums, or unions
//~| ERROR the size for values of type `(dyn Iterator<Item = impl Sized> + 'static)`

fn main() {}
