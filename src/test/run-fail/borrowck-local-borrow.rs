// error-pattern:panic 1

// revisions: migrate mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let x = 2;
    let y = &x;
    panic!("panic 1");
}
