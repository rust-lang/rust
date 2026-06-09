impl ! {} // OK

default unsafe FAIL //~ ERROR expected item, found keyword `unsafe`
//~^ ERROR `default` is not followed by an item
