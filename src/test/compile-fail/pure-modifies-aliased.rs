// Check that pure functions cannot modify aliased state.

pure fn modify_in_ref(&&sum: {mut f: int}) {
    sum.f = 3; //! ERROR assigning to mutable field prohibited in pure functions
}

pure fn modify_in_box(sum: @mut {f: int}) {
    sum.f = 3; //! ERROR assigning to mutable field prohibited in pure functions
}

impl foo for int {
    pure fn modify_in_box_rec(sum: @{mut f: int}) {
        sum.f = self; //! ERROR assigning to mutable field prohibited in pure functions
    }
}

fn main() {
}