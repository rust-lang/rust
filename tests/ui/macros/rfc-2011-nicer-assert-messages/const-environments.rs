#![feature(core_intrinsics, generic_assert)]

const _: () = {
    let foo = 1;
    assert!(foo == 3);
    //~^ERROR: evaluation of constant value failed
};

fn main() {
    const {
        let foo = 1;
        assert!(foo == 3);
    }

    const fn bar() {
        let foo = 1;
        assert!(foo == 3);
    }
}
