type Arr<const N: usize> = [u8; N - 1];
//~^ ERROR generic parameters may not be used in const operations

fn test<const N: usize>() -> Arr<N> where Arr<N>: Default {
    Default::default()
}

fn main() {
    let x = test::<33>();
    assert_eq!(x, [0; 32]);
}
