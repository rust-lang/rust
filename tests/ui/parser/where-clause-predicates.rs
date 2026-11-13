// Exercise the parsing of the "bounds" (predicates) that are found in where clauses.
//
// Check that we allow zero bounds after `:`, zero predicates after `where`,
// trailing `,` in predicate lists & trailing `+` in bound lists.
//
//@ compile-flags: -Zparse-crate-root-only

type A where for<'a> for<'b> Trait1 + ?Trait2: 'a + Trait = u8; // OK
type A where T: Trait, = u8; // OK
type A where T: = u8; // OK
type A where T:, = u8; // OK
type A where T: Trait + Trait = u8; // OK
type A where = u8; // OK
type A where T: Trait + = u8; // OK
fn barrier() { type A where T, = u8; }
//~^ ERROR expected one of `!`, `(`, `+`, `::`, `:`, `<`, `==`, or `=`, found `,`

type A where 'a: 'b + 'c = u8; // OK
type A where 'a: 'b, = u8; // OK
type A where 'a: = u8; // OK
type A where 'a:, = u8; // OK
type A where 'a: 'b + 'c = u8; // OK
type A where = u8; // OK
type A where 'a: 'b + = u8; // OK
fn barrier() { type A where , = u8; }
//~^ ERROR expected one of `#`, `;`, `=`, `where`, lifetime, or type, found `,`
