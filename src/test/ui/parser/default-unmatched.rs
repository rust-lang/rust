mod foo {
    default!(); // OK.
    default do
    //~^ ERROR `default` not followed by an item
    //~| ERROR expected item, found reserved keyword `do`
}
