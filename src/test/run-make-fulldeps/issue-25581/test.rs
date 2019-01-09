#[link(name = "test", kind = "static")]
extern {
    fn slice_len(s: &[u8]) -> usize;
    fn slice_elem(s: &[u8], idx: usize) -> u8;
}

fn main() {
    let data = [1,2,3,4,5];

    unsafe {
        assert_eq!(data.len(), slice_len(&data) as usize);
        assert_eq!(data[0], slice_elem(&data, 0));
        assert_eq!(data[1], slice_elem(&data, 1));
        assert_eq!(data[2], slice_elem(&data, 2));
        assert_eq!(data[3], slice_elem(&data, 3));
        assert_eq!(data[4], slice_elem(&data, 4));
    }
}
