#![crate_type="lib"]

extern "C" {
    // @has lib/fn.foreigner.html //pre 'pub unsafe fn foreigner(cold_as_ice: u32)'
    pub fn foreigner(cold_as_ice: u32);
}
