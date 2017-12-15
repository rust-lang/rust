fn main() {
    let data: [u8; 1024*1024*1024] = [42; 1024*1024*1024];
    //~^ ERROR: tried to allocate
    assert_eq!(data.len(), 1024*1024*1024);
}
