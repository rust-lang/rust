// run-pass

trait SignedUnsigned {
    type Opposite;
    fn convert(self) -> Self::Opposite;
}

impl SignedUnsigned for isize {
    type Opposite = usize;

    fn convert(self) -> usize {
        self as usize
    }
}

impl SignedUnsigned for usize {
    type Opposite = isize;

    fn convert(self) -> isize {
        self as isize
    }
}

fn get(x: isize) -> <isize as SignedUnsigned>::Opposite {
    x.convert()
}

fn main() {
    let x = get(22);
    assert_eq!(22, x);
}
