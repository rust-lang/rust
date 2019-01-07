type A where for<'a> for<'b> Trait1 + ?Trait2: 'a + Trait = u8; // OK
type A where T: Trait, = u8; // OK
type A where T: = u8; // OK
type A where T:, = u8; // OK
type A where T: Trait + Trait = u8; // OK
type A where = u8; // OK
type A where T: Trait + = u8; // OK
type A where T, = u8;
//~^ ERROR expected one of `!`, `(`, `+`, `::`, `:`, `<`, `==`, or `=`, found `,`

fn main() {}
