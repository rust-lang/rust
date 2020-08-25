enum E{A((?'a a+?+l))}
//~^ ERROR `?` may only modify trait bounds, not lifetime bounds
//~| ERROR expected one of `)`, `+`, or `,`
//~| ERROR expected identifier, found `)`
//~| ERROR cannot find trait
//~| ERROR use of undeclared lifetime
//~| WARN trait objects without

fn main() {}
