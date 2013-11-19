pub fn main() {
    // FIXME(#2202) - Due to the way that borrowck treats closures,
    // you get two error reports here.
    let bar = ~3;
    let _g = || { //~ ERROR capture of moved value
        let _h: proc() -> int = || *bar; //~ ERROR capture of moved value
    };
}
