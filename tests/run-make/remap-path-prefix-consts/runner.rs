// Verifies that the paths are the same and consistent between this crate and location_caller crate.
//
// https://github.com/rust-lang/rust/issues/148328

extern crate location_caller;

fn main() {
    {
        // Assert both paths are the same
        let the_path = location_caller::the_path();
        let the_path2 = location_caller::the_path2();
        assert_eq!(the_path, the_path2);
    }

    {
        // Let's make sure we don't read OOB memory
        println!("{:?}", location_caller::the_zeroed_path_len_array());
    }
}
