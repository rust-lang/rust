//! What exactly is allowed on stable is a bit strange and arbitrary. This tests various
//! combinations of parens and braces to make sure they remain stable.

struct Thing<const N: usize>;

fn f<const N: usize>() {
    let _: [u32; _] = [5; 5];
    let _: [u32; (_)] = [5; 5];
    let _: [u32; { _ }] = [5; 5]; //~ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    let _: [u32; { (_) }] = [5; 5]; //~ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    let _: [u32; N] = [5; _];
    let _: [u32; (N)] = [5; _]; //~ ERROR generic parameters may not be used in const operations
    let _: [u32; { N }] = [5; _];
    let _: [u32; { (N) }] = [5; _]; //~ ERROR generic parameters may not be used in const operations
    let _: [u32; { { N } }] = [5; _]; //~ ERROR generic parameters may not be used in const operations
    let _: Thing<_> = Thing::<5>;
    let _: Thing<(_)> = Thing::<5>;
    let _: Thing<{ _ }> = Thing::<5>; //~ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    let _: Thing<{ (_) }> = Thing::<5>; //~ ERROR in expressions, `_` can only be used on the left-hand side of an assignment
    let _: Thing<N> = Thing;
    let _: Thing<(N)> = Thing; //~ ERROR cannot find type `N` in this scope
    //~| ERROR unresolved item provided when a constant was expected
    let _: Thing<{ N }> = Thing;
    let _: Thing<{ (N) }> = Thing; //~ ERROR generic parameters may not be used in const operations
    let _: Thing<{ { N } }> = Thing; //~ ERROR generic parameters may not be used in const operations
}

fn main() {}
