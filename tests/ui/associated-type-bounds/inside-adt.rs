use std::mem::ManuallyDrop;

struct S1 { f: dyn Iterator<Item: Copy> }
//~^ ERROR associated type bounds are not allowed in `dyn` types
struct S2 { f: Box<dyn Iterator<Item: Copy>> }
//~^ ERROR associated type bounds are not allowed in `dyn` types
struct S3 { f: dyn Iterator<Item: 'static> }
//~^ ERROR associated type bounds are not allowed in `dyn` types

enum E1 { V(dyn Iterator<Item: Copy>) }
//~^ ERROR associated type bounds are not allowed in `dyn` types
enum E2 { V(Box<dyn Iterator<Item: Copy>>) }
//~^ ERROR associated type bounds are not allowed in `dyn` types
enum E3 { V(dyn Iterator<Item: 'static>) }
//~^ ERROR associated type bounds are not allowed in `dyn` types

union U1 { f: ManuallyDrop<dyn Iterator<Item: Copy>> }
//~^ ERROR associated type bounds are not allowed in `dyn` types
union U2 { f: ManuallyDrop<Box<dyn Iterator<Item: Copy>>> }
//~^ ERROR associated type bounds are not allowed in `dyn` types
union U3 { f: ManuallyDrop<dyn Iterator<Item: 'static>> }
//~^ ERROR associated type bounds are not allowed in `dyn` types

fn main() {}
