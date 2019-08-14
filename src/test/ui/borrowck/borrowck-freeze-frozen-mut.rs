// run-pass
// Test that a `&mut` inside of an `&` is freezable.


struct MutSlice<'a, T:'a> {
    data: &'a mut [T]
}

fn get<'a, T>(ms: &'a MutSlice<'a, T>, index: usize) -> &'a T {
    &ms.data[index]
}

pub fn main() {
    let mut data = [1, 2, 3];
    {
        let slice = MutSlice { data: &mut data };
        slice.data[0] += 4;
        let index0 = get(&slice, 0);
        let index1 = get(&slice, 1);
        let index2 = get(&slice, 2);
        assert_eq!(*index0, 5);
        assert_eq!(*index1, 2);
        assert_eq!(*index2, 3);
    }
    assert_eq!(data[0], 5);
    assert_eq!(data[1], 2);
    assert_eq!(data[2], 3);
}
