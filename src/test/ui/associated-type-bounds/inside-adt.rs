// compile-fail
// ignore-tidy-linelength
// error-pattern:could not find defining uses

#![feature(associated_type_bounds)]
#![feature(untagged_unions)]

struct S1 { f: dyn Iterator<Item: Copy> }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
struct S2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
struct S3 { f: dyn Iterator<Item: 'static> }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]

enum E1 { V(dyn Iterator<Item: Copy>) }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
enum E2 { V(Box<dyn Iterator<Item: Copy>>) }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
enum E3 { V(dyn Iterator<Item: 'static>) }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]

union U1 { f: dyn Iterator<Item: Copy> }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
union U2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
union U3 { f: dyn Iterator<Item: 'static> }
//~^ associated type bounds are not allowed within structs, enums, or unions
//~| the value of the associated type `Item` (from the trait `std::iter::Iterator`) must be specified [E0191]
