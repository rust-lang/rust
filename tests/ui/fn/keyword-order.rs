//@ edition:2018

default pub const async unsafe extern fn err() {} //~ ERROR `default` is not followed by an item
//~^ ERROR expected item, found keyword `pub`

pub default const async unsafe extern fn ok() {}
