//@ run-pass
//@ revisions: default mir-opt
//@[mir-opt] compile-flags: -Zmir-opt-level=4

fn pass_to_ptr_call<T>(f: fn(T), x: T) {
    f(x);
}

#[track_caller]
fn tracked_unit(_: ()) {
    let expected_line = line!() - 1;
    let location = std::panic::Location::caller();
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
}

trait Trait {
    fn trait_tracked_unit(_: ());
}

impl Trait for () {
    #[track_caller]
    fn trait_tracked_unit(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
    }
}

trait TrackedTrait {
    #[track_caller]
    fn trait_tracked_unit_default(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
    }
}

impl TrackedTrait for () {}

trait BlanketTrackedTrait {
    #[track_caller]
    fn tracked_blanket(_: ());
}

impl BlanketTrackedTrait for () {
    fn tracked_blanket(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
    }
}

fn main() {
    pass_to_ptr_call(tracked_unit, ());
    pass_to_ptr_call(<() as Trait>::trait_tracked_unit, ());
    pass_to_ptr_call(<() as TrackedTrait>::trait_tracked_unit_default, ());
    pass_to_ptr_call(<() as BlanketTrackedTrait>::tracked_blanket, ());
}
