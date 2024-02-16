//@ run-pass

macro_rules! assert_expansion_site_is_tracked {
    () => {{
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_ne!(location.line(), line!(), "line should be outside this fn");
    }}
}

trait Tracked {
    fn local_tracked(&self);

    #[track_caller]
    fn blanket_tracked(&self);

    #[track_caller]
    fn default_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }
}

impl Tracked for () {
    #[track_caller]
    fn local_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }

    fn blanket_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }
}

impl Tracked for bool {
    #[track_caller]
    fn local_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }

    fn blanket_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }

    fn default_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }
}

impl Tracked for u8 {
    #[track_caller]
    fn local_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }

    fn blanket_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }

    #[track_caller]
    fn default_tracked(&self) {
        assert_expansion_site_is_tracked!();
    }
}

fn main() {
    ().local_tracked();
    ().default_tracked();
    ().blanket_tracked();

    true.local_tracked();
    true.default_tracked();
    true.blanket_tracked();

    0u8.local_tracked();
    0u8.default_tracked();
    0u8.blanket_tracked();
}
