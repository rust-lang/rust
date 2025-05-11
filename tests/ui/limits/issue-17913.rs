//@ build-fail
//@ normalize-stderr: "\[&usize; \d+\]" -> "[&usize; usize::MAX]"

#[cfg(target_pointer_width = "64")]
fn main() {
    let n = 0_usize;
    let a: Box<_> = Box::new([&n; 0xF000000000000000_usize]);
    println!("{}", a[0xFFFFFF_usize]);
}

#[cfg(target_pointer_width = "32")]
fn main() {
    let n = 0_usize;
    let a: Box<_> = Box::new([&n; 0xFFFFFFFF_usize]);
    println!("{}", a[0xFFFFFF_usize]);
}

//~? ERROR are too big for the target architecture
