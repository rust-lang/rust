//@ run-pass

trait Tracked {
    #[track_caller]
    fn track_caller_trait_method(&self, line: u32, col: u32) {
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        // The trait method definition is annotated with `#[track_caller]`,
        // so caller location information will work through a method
        // call on a trait object
        assert_eq!(location.line(), line, "Bad line");
        assert_eq!(location.column(), col, "Bad col");
    }

    fn track_caller_not_on_trait_method(&self);

    #[track_caller]
    fn track_caller_through_self(self: Box<Self>, line: u32, col: u32);
}

impl Tracked for () {
    // We have `#[track_caller]` on the implementation of the method,
    // but not on the definition of the method in the trait. Therefore,
    // caller location information will *not* work through a method call
    // on a trait object. Instead, we will get the location of this method
    #[track_caller]
    fn track_caller_not_on_trait_method(&self) {
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_eq!(location.line(), line!() - 3);
        assert_eq!(location.column(), 5);
    }

    // We don't have a `#[track_caller]` attribute, but
    // `#[track_caller]` is present on the trait definition,
    // so we'll still get location information
    fn track_caller_through_self(self: Box<Self>, line: u32, col: u32) {
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        // The trait method definition is annotated with `#[track_caller]`,
        // so caller location information will work through a method
        // call on a trait object
        assert_eq!(location.line(), line, "Bad line");
        assert_eq!(location.column(), col, "Bad col");
    }
}

fn main() {
    let tracked: &dyn Tracked = &();
    // The column is the start of 'track_caller_trait_method'
    tracked.track_caller_trait_method(line!(), 13);

    const TRACKED: &dyn Tracked = &();
    // The column is the start of 'track_caller_trait_method'
    TRACKED.track_caller_trait_method(line!(), 13);
    TRACKED.track_caller_not_on_trait_method();

    // The column is the start of `track_caller_through_self`
    let boxed: Box<dyn Tracked> = Box::new(());
    boxed.track_caller_through_self(line!(), 11);
}
