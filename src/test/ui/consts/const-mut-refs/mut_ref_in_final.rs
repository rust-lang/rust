#![feature(const_mut_refs)]
#![feature(const_fn)]
#![feature(raw_ref_op)]
const NULL: *mut i32 = std::ptr::null_mut();
const A: *const i32 = &4;

// It could be made sound to allow it to compile,
// but we do not want to allow this to compile,
// as that would be an enormous footgun in oli-obk's opinion.
const B: *mut i32 = &mut 4; //~ ERROR mutable references are not allowed

// Could be ok, but the same analysis that prevents the mutable one above will also bail out here
// Using a block with some complex content, because just `&45` would get promoted,
// which is not what we want to test here.
const C: *const i32 = &{
    let mut x = 42;
    x += 3;
    x
};

fn main() {
    println!("{}", unsafe { *A });
    unsafe { *B = 4 } // Bad news
}
