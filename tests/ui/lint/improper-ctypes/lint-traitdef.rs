#![deny(improper_ctypes, improper_ctypes_definitions)]
#![feature(associated_type_defaults)]

/// This test ensures that all the necessary elements of trait/impl definitions are being linted
/// for FFI-safety

trait Bass {
    const IS_IT_A_FISH_OR_AN_INSTRUMENT: bool;
    const OKAY_BUT_SERIOUSLY: extern "C" fn() -> &'static str;  //~ ERROR: uses type `&str`
    type Group: ::std::ops::Add<Output=Self::Group> + Sized;
    type MetaGroup = extern "C" fn()->(u8,Self::Group);  //~ ERROR: uses type `(u8, <Self as Bass>::Group)`

    extern "C" fn count_strings() -> u32 {
        4
    }
    extern "C" fn string_names() -> &'static [&'static str] {
        //~^ ERROR: uses type `&[&str]`
        &["E", "A", "D", "G"]
    }


    extern "C" fn breathe_water(&mut self);
    extern "C" fn eat_food(&mut self, food_name: &str); //~ ERROR: uses type `&str`

    extern "C" fn into_solo(&self) -> Self::Group;
    extern "C" fn group_up(&self, g: <Self as Bass>::Group) -> <Self as Bass>::Group {
        g + self.into_solo()
    }
}


extern "C" fn fishy_bass_inner_answer() -> &'static str {
    //~^ ERROR: uses type `&str`
    "both"
}

struct FishyBass {}
impl Bass for FishyBass {
    const IS_IT_A_FISH_OR_AN_INSTRUMENT: bool = true;
    const OKAY_BUT_SERIOUSLY: extern "C" fn() -> &'static str = fishy_bass_inner_answer;
    //~^ ERROR: uses type `&str`
    type Group = u32;

    extern "C" fn breathe_water(&mut self) {}
    extern "C" fn eat_food(&mut self, food_name: &str) {}  //~ ERROR: uses type `&str`
    extern "C" fn into_solo(&self) -> Self::Group {1_u32}
}

fn main(){}
