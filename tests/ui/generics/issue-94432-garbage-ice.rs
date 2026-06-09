//@ check-fail
//@ dont-check-compiler-stdout
//@ dont-check-compiler-stderr

fn�a<e>(){fn�p(){e}} //~ ERROR unknown start of token: \u{fffd}
//~^ ERROR unknown start of token: \u{fffd}
//~^^ ERROR can't use generic parameters from outer item [E0401]
//~^^^ WARN type parameter `e` should have an upper camel case name

fn main(){}
