// run-pass

trait Tracked {
    #[track_caller]
    fn handle(&self) {
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        // we only call this via trait object, so the def site should *always* be returned
        assert_eq!(location.line(), line!() - 4);
        assert_eq!(location.column(), 5);
    }
}

impl Tracked for () {}
impl Tracked for u8 {}

fn main() {
    let tracked: &dyn Tracked = &5u8;
    tracked.handle();

    const TRACKED: &dyn Tracked = &();
    TRACKED.handle();
}
