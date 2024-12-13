//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
mod safe {
    use std::slice::from_raw_parts_mut;

    pub fn as_mut_slice<T>(self_: &Vec<T>) -> &mut [T] {
        unsafe { from_raw_parts_mut(self_.as_ptr() as *mut T, self_.len()) }
    }
}

fn main() {
    let v = vec![0, 1, 2];
    let v1 = safe::as_mut_slice(&v);
    let v2 = safe::as_mut_slice(&v);
    v1[1] = 5;
    //~[stack]^ ERROR: /write access .* tag does not exist in the borrow stack/
    v2[1] = 7;
    //~[tree]^ ERROR: /write access through .* is forbidden/
}
