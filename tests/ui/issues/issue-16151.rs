// run-pass

use std::mem;

static mut DROP_COUNT: usize = 0;

struct Fragment;

impl Drop for Fragment {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1;
        }
    }
}

fn main() {
    {
        let mut fragments = vec![Fragment, Fragment, Fragment];
        let _new_fragments: Vec<Fragment> = mem::replace(&mut fragments, vec![])
            .into_iter()
            .skip_while(|_fragment| {
                true
            }).collect();
    }
    unsafe {
        assert_eq!(DROP_COUNT, 3);
    }
}
