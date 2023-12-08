// aux-crate:bevy_ecs=bevy_ecs.rs
// check-pass
// Related to Bevy regression #118553

extern crate bevy_ecs;

use bevy_ecs::*;

fn handler<'a>(_: ParamSet<Query<&'a u8>>) {}

fn main() {}
