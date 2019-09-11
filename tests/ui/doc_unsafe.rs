/// This is not sufficiently documented
pub unsafe fn destroy_the_planet() {
    unimplemented!();
}

/// This one is
///
/// # Safety
///
/// This function shouldn't be called unless the horsemen are ready
pub unsafe fn apocalypse(universe: &mut ()) {
    unimplemented!();
}

/// This is a private function, so docs aren't necessary
unsafe fn you_dont_see_me() {
    unimplemented!();
}

fn main() {
    you_dont_see_me();
    destroy_the_planet();
    let mut universe = ();
    apocalypse(&mut universe);
}
