// Test different scenarios with impossible adjustments due to blocks
// or no-op target expressions.

#![feature(fn_delegation)]

trait Trait: Sized {
    fn by_value(self) -> i32 { 1 }
    fn by_mut_ref(&mut self) -> i32 { 2 }
    fn by_ref(&self) -> i32 { 3 }
}

struct F;
impl Trait for F {}

struct Struct(F);
reuse impl Trait for Struct { self.0 }

struct S(F);
reuse impl Trait for S { { self.0 } }
//~^ ERROR: cannot move out of `self` which is behind a shared reference
//~| ERROR: cannot move out of `self` which is behind a mutable reference

struct S1(F);
reuse impl Trait for S1 { { { { { { self.0 } } } } } }
//~^ ERROR: cannot move out of `self` which is behind a shared reference
//~| ERROR: cannot move out of `self` which is behind a mutable reference

struct S2(F);
reuse impl Trait for S2 { }
//~^ WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]

struct S3(F);
reuse impl Trait for S3 { (); }
//~^ WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]

struct S4(F);
reuse impl Trait for S4 { println!(); }
//~^ WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]

struct S5(F);
reuse impl Trait for S5 { fn foo() {} }
//~^ WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]

struct S6(F);
reuse impl Trait for S6;
//~^ WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]
//~| WARN: function cannot return without recursing [unconditional_recursion]

fn main() {}
