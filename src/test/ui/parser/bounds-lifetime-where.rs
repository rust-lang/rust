type A where 'a: 'b + 'c = u8; // OK
type A where 'a: 'b, = u8; // OK
type A where 'a: = u8; // OK
type A where 'a:, = u8; // OK
type A where 'a: 'b + 'c = u8; // OK
type A where = u8; // OK
type A where 'a: 'b + = u8; // OK
type A where , = u8; //~ ERROR expected one of `=`, lifetime, or type, found `,`

fn main() {}
