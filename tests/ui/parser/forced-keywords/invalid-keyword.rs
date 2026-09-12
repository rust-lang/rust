//@ edition: 2021..
#![feature(forced_keywords)]

const _: () = k#not_a_keyword();
//~^ ERROR `not_a_keyword` is not a valid keyword
//~| ERROR expected expression, found `k#not_a_keyword`
