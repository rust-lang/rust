fn main() {
    [0; 3][3u64 as usize]; //~ ERROR the len is 3 but the index is 3
}
