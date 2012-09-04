fn main() {

    let x = {
        f: 0,
        g: 0,
    };

    let y = {
        f: 1,
        g: 1,
        .. x
    };

    let z = {
        f: 1,
        .. x
    };

}
