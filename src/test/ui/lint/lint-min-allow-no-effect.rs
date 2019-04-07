#![deny(unused_attributes)]
//~^ NOTE lint level defined here

#![allow(illegal_floating_point_literal_pattern)]
//~^ ERROR #[allow(illegal_floating_point_literal_pattern)] has no effect [unused_attributes]
//~| NOTE the minimum lint level for `illegal_floating_point_literal_pattern` is `warn`
//~| NOTE the lint level cannot be reduced to `allow`
//~| HELP remove the #[allow(illegal_floating_point_literal_pattern)] directive
//~| WARN `illegal_floating_point_literal_pattern` was previously accepted
//~| WARN hard error
//~| NOTE for more information
//~| ERROR #[allow(illegal_floating_point_literal_pattern)] has no effect [unused_attributes]
//~| NOTE the minimum lint level for `illegal_floating_point_literal_pattern` is `warn`
//~| NOTE the lint level cannot be reduced to `allow`
//~| HELP remove the #[allow(illegal_floating_point_literal_pattern)] directive
//~| WARN `illegal_floating_point_literal_pattern` was previously accepted
//~| WARN hard error
//~| NOTE for more information

fn main() {}
