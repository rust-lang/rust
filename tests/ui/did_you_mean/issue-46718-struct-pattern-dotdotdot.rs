#![allow(unused)]

struct PersonalityInventory {
    expressivity: f32,
    instrumentality: f32
}

impl PersonalityInventory {
    fn expressivity(&self) -> f32 {
        match *self {
            PersonalityInventory { expressivity: exp, ... } => exp
            //~^ ERROR expected field pattern, found `...`
        }
    }
}

fn main() {}
