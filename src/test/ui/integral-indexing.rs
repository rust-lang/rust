pub fn main() {
    let v: Vec<isize> = vec![0, 1, 2, 3, 4, 5];
    let s: String = "abcdef".to_string();
    v[3_usize];
    v[3];
    v[3u8];  //~ERROR : the type `[isize]` cannot be indexed by `u8`
    v[3i8];  //~ERROR : the type `[isize]` cannot be indexed by `i8`
    v[3u32]; //~ERROR : the type `[isize]` cannot be indexed by `u32`
    v[3i32]; //~ERROR : the type `[isize]` cannot be indexed by `i32`
    s.as_bytes()[3_usize];
    s.as_bytes()[3];
    s.as_bytes()[3u8];  //~ERROR : the type `[u8]` cannot be indexed by `u8`
    s.as_bytes()[3i8];  //~ERROR : the type `[u8]` cannot be indexed by `i8`
    s.as_bytes()[3u32]; //~ERROR : the type `[u8]` cannot be indexed by `u32`
    s.as_bytes()[3i32]; //~ERROR : the type `[u8]` cannot be indexed by `i32`
}
