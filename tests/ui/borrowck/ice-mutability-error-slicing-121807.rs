//@ edition:2015
// test for ICE #121807 begin <= end (12 <= 11) when slicing 'Self::Assoc<'_>'
// fixed by #122749

trait MemoryUnit { // ERROR: not all trait items implemented, missing: `read_word`
    extern "C" fn read_word(&mut self) -> u8;
    extern "C" fn read_dword(Self::Assoc<'_>) -> u16;
    //~^ WARN anonymous parameters are deprecated and will be removed in the next edition
    //~^^ WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
    //~^^^ ERROR associated type `Assoc` not found for `Self`
}

struct ROM {}

impl MemoryUnit for ROM {
//~^ ERROR not all trait items implemented, missing: `read_word`
    extern "C" fn read_dword(&'s self) -> u16 {
    //~^ ERROR use of undeclared lifetime name `'s`
    //~^^ ERROR method `read_dword` has a `&self` declaration in the impl, but not in the trait
        let a16 = self.read_word() as u16;
        let b16 = self.read_word() as u16;

        (b16 << 8) | a16
    }
}

pub fn main() {}
