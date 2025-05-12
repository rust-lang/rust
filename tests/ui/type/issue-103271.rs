fn main() {
    let iter_fun = <&[u32]>::iter;
    //~^ ERROR no function or associated item named `iter` found for reference `&[u32]` in the current scope [E0599]
    //~| NOTE function or associated item not found in `&[u32]`
    //~| HELP the function `iter` is implemented on `[u32]`
    for item in iter_fun(&[1,1]) {
        let x: &u32 = item;
        assert_eq!(x, &1);
    }
    let iter_fun2 = <(&[u32])>::iter;
    //~^ ERROR no function or associated item named `iter` found for reference `&[u32]` in the current scope [E0599]
    //~| NOTE function or associated item not found in `&[u32]`
    //~| HELP the function `iter` is implemented on `[u32]`
    for item2 in iter_fun2(&[1,1]) {
        let x: &u32 = item2;
        assert_eq!(x, &1);
    }
}
