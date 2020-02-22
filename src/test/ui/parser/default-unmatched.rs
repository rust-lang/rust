mod foo {
    default!(); // OK.
    default do
    //~^ ERROR unmatched `default`
    //~| ERROR expected item, found reserved keyword `do`
}
