use std::libc::size_t;


fn foo(x: size_t) {
    println!("{}", x);
}

fn main() {
    let x: u64 = 1;
    
    // on 64 bit linux, size_t is u64, therefore the cast seems unnecessary,
    // but on 32 bit linux is's u32, so the cast is needed
    foo(x as size_t);
}
