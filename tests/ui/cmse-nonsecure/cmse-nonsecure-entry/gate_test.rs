// gate-test-cmse_nonsecure_entry

#[no_mangle]
pub extern "C-cmse-nonsecure-entry" fn entry_function(input: u32) -> u32 {
    //~^ ERROR [E0570]
    //~| ERROR [E0658]
    input + 6
}

fn main() {}
