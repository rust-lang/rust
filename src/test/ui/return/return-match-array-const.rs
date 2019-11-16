fn main() {
    [(); return match 0 { n => n }];
    //~^ ERROR: return statement outside of function body
    //~| ERROR: `match` is not allowed in a `const`

    [(); return match 0 { 0 => 0 }];
    //~^ ERROR: return statement outside of function body
    //~| ERROR: `match` is not allowed in a `const`

    [(); return match () { 'a' => 0, _ => 0 }];
    //~^ ERROR: return statement outside of function body
    //~| ERROR: `match` is not allowed in a `const`
}
