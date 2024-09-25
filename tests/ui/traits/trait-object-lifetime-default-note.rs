trait A {}

impl<T> A for T {}

fn main() {
    let local = 0; //~ NOTE binding `local` declared here
    let r = &local; //~ ERROR `local` does not live long enough
    //~| NOTE borrowed value does not live long enough
    //~| NOTE due to object lifetime defaults, `Box<dyn A>` actually means `Box<(dyn A + 'static)>`
    require_box(Box::new(r));
    //~^ NOTE coercion requires that `local` is borrowed for `'static`

    let _ = 0;
} //~ NOTE `local` dropped here while still borrowed

fn require_box(_a: Box<dyn A>) {}
