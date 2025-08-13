// Test case for issue #134061 - exact reproduction case
// Tests the specific case where EOF is encountered after lifetime in pattern

const x: () = |&'a
//~^ ERROR unexpected lifetime `'a` in pattern
//~| HELP remove the lifetime
//~| ERROR expected parameter name, found `<eof>`
