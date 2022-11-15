fn main() {
    let iter_fun = <&[u32]>::iter;
    //~^ ERROR no function or associated item named `iter` found for reference `&[u32]` in the current scope [E0599]
    //~| function or associated item not found in `&[u32]`
    //~| HELP the function `iter` is implemented on `[u32]`
    for item in iter_fun(&[1,1]) {
        let x: &u32 = item;
        assert_eq!(x, &1);
    }
}
